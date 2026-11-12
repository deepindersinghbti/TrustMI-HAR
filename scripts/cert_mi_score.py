"""Plan step 5: does mutual information add to the certificate? Scored exactly like the paper's numbers.

  python scripts/cert_mi_score.py --dataset d1        (UTD-MHAD: B4 and TMC as main model)
  python scripts/cert_mi_score.py --dataset d2        (MMAct:    B4 and TMC-s)
  python scripts/cert_mi_score.py --dataset d1v2      (journal: UTD-MHAD, whole-frame video; B4, TMC, ECML)
  python scripts/cert_mi_score.py --dataset d2v       (journal: MMAct with video, 3 sensors; B4s3, TMC, ECML)
Journal datasets: three sensors each, so naming and use / drop / abstain are scored on both; there are no stored
paper numbers yet, so the check is against cert_policy.evaluate instead. Needs configs/backups.json entries
(scripts/choose_backups_journal.py).
No existing scorer is edited: this script IMPORTS cert_evidential_score.per_subject (the per-test-subject
loop behind every number in the paper: logistic score fitted on the validation person, naming rule chosen on
validation, use/drop/abstain revisions 1 and 2) and only gives it extra inputs from cert_mi.py.

CHECK FIRST: the current certificate is recomputed here and must equal the stored per-subject numbers of
the paper (e_<ds>_per_subject.csv, e_<ds>_on_<tmc>_per_subject.csv) to 1e-9, or the script stops.

Certificate versions (all label-free signals; every weight, threshold and rule fitted on validation):
  C3     current paper certificate: s1 = 1 - msp, s2 = witness JS, s3 = 1 - lowest witness support
  MI     s4 only (= - lowest pairwise pointwise MI)                      naming: lowest MI share
  s1+MI  the fused model's doubt + MI, no witnesses                     naming: lowest MI share
  REV3   s1, s2, s3, s4  [MAIN, fixed in the plan before any result]   naming: odd one out OR lowest MI share
  REV3f  as REV3 with the frozen-encoder matcher (ablation)
Also: per-sensor lie detection (UTD-MHAD), and the dataset-level MI lower bound per sensor pair (validation).
Writes results/certificate/mi_<ds>_per_subject.csv, mi_<ds>_on_<tmc>_per_subject.csv,
       mi_<ds>_per_sensor.csv, mi_bound_summary.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cert_policy as CP
from cert_evidential_score import per_subject, ev_feats, ev_drop, ci, EV_RULES, EVFILE, RES, fit_apply, can_name
from cert_score import auroc, fmt, tq, unit

KEYS = ["fold", "part", "seed", "clip", "condition"]
MIFILE = {"d1": "mi_d1_{v}_all_s0-1-2_att-imu-skel-video", "d2": "mi_d2_{v}_all_s0-1-2"}
EXPFILE = {"d1": "experts_d1_all_s0-1-2_att-imu-skel-video", "d2": "experts_d2_all_s0-1-2"}
MAIN_EV = {"d1": ("tmc", "ecml"), "d2": ("tmc_s", "ecml_s"),      # the versions the paper reports
           "d1v2": ("tmc", "ecml"), "d2v": ("tmc_s3", "ecml_s3"), "d3": ("tmc_w", "ecml_w")}  # journal
CHECK_COLS = ["lie_auroc", "naming", "acc_policy_lie", "acc_policy_normal", "all_cov80_cert_policy", "error_auroc"]

ODD = ("exp_div", "max")
SYSTEMS = {   # name -> (signals, naming rules)
    "C3 (paper certificate)": (["s1", "s2", "s3"], {"odd one out": ODD}),
    "MI only": (["s4"], {"lowest MI share": ("mi_share", "min")}),
    "s1 + MI (no witnesses)": (["s1", "s4"], {"lowest MI share": ("mi_share", "min")}),
    "REV3 = C3 + MI (main)": (["s1", "s2", "s3", "s4"], {"odd one out": ODD, "lowest MI share": ("mi_share", "min")}),
    "REV3 frozen MI (ablation)": (["s1", "s2", "s3", "fz_s4"],
                                  {"odd one out": ODD, "lowest MI share": ("fz_mi_share", "min")}),
}
MAIN, REF = "REV3 = C3 + MI (main)", "C3 (paper certificate)"


def signals(names):
    def feats(d, ds):
        c3 = CP.features(d, ds)                                     # exactly the paper's s1, s2, s3
        cols = {"s1": c3[:, 0], "s2": c3[:, 1], "s3": c3[:, 2]}
        return np.column_stack([cols[n] if n in cols else d[n].to_numpy() for n in names])
    return feats


def add_mi(df, ds):
    """Merge both matcher versions' MI columns onto df (one row each, nothing lost)."""
    for v, pre in (("own", ""), ("frozen", "fz_")):
        mi = pd.read_parquet(RES / f"{MIFILE[ds].format(v=v)}.parquet")
        cols = [c for c in mi.columns if c.startswith(("pmi_", "mi_", "s4"))]
        mi = mi[KEYS + cols].rename(columns={c: pre + c for c in cols})
        n = len(df)
        df = df.merge(mi, on=KEYS, how="inner", validate="one_to_one")
        if len(df) != n:
            raise SystemExit(f"{ds} {v}: {n - len(df)} rows have no MI row: stop and paste this")
    return df


def journal_files(ds):
    """Journal datasets: file names follow queue C (no --attack suffix); CP.TAG selects all folds or a smoke run."""
    MIFILE[ds] = f"mi_{ds}_{{v}}_{CP.TAG}"
    EXPFILE[ds] = f"experts_{ds}_{CP.TAG}"
    EVFILE[ds] = f"evidential_{ds}_{CP.TAG}"


def check_policy(per, df, ds, label):
    """Journal datasets have no stored paper numbers yet. Instead the C3 row computed here must equal an
    independent computation of the same policy by cert_policy.evaluate (revisions 1 and 2)."""
    worst = 0.0
    for rev in (1, 2):
        CP.REV = rev
        ref = CP.evaluate(df, ds, "odd", "check").set_index("subject")
        a = per[(REF, rev)]
        worst = max(worst, *(float(np.nanmax(np.abs(a[c] - ref.loc[a.index, c])))
                             for c in ("acc_policy_lie", "acc_policy_normal", "all_cov80_cert_policy")))
    CP.REV = 1
    if worst > 1e-9:
        raise SystemExit(f"{label}: C3 differs from cert_policy.evaluate by {worst:.2e}: stop and paste this")
    print(f"check ({label}): C3 equals cert_policy.evaluate exactly (largest difference {worst:.1e})")


def check(per, stored, label):
    worst = 0.0
    for rev in (1, 2):
        a = per[(REF, rev)]
        b = stored[stored.revision == rev].set_index("subject").loc[a.index]
        for c in CHECK_COLS:
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                raise SystemExit(f"{label}: NaN pattern of {c} differs from the paper's file: stop and paste this")
            worst = max(worst, float(np.nanmax(np.abs(x - y))) if (~np.isnan(x)).any() else 0.0)
    if worst > 1e-9:
        raise SystemExit(f"{label}: C3 differs from the paper's stored numbers by {worst:.2e}: stop and paste this")
    print(f"check ({label}): the paper's certificate is reproduced exactly (largest difference {worst:.1e})")


def score_all(df, ds, drop_answer):
    per = {}
    for name, (sig, rules) in SYSTEMS.items():
        for rev in (1, 2):
            per[(name, rev)] = per_subject(df, ds, signals(sig), rules, drop_answer, rev)
    return per


def table(per, ds, others, title):
    sysnames = list(dict.fromkeys(k[0] for k in per))
    for rev in (1, 2):
        print(f"\n{ds}: {title}, revision {rev}" + (" (post hoc, labelled)" if rev == 2 else ""))
        print(f"  {'system':30s} {'lie_AUROC':>9s} {'error_AUROC':>11s} {'naming%':>8s} {'lie_acc%':>8s} {'normal%':>8s} {'ans80%':>7s}  rules chosen")
        for s in sysnames:
            q = per[(s, rev)]
            nm = f"{100 * q.naming.mean():8.1f}" if can_name(ds) else "     n/a"
            rules = q.rule.value_counts().to_dict() if can_name(ds) and s not in others else ""
            print(f"  {s:30s} {q.lie_auroc.mean():9.3f} {q.error_auroc.mean():11.3f} {nm} "
                  f"{100 * q.acc_policy_lie.mean():8.1f} {100 * q.acc_policy_normal.mean():8.1f} "
                  f"{100 * q.all_cov80_cert_policy.mean():7.1f}  {rules}")
        o = per[(MAIN, rev)]
        for s in [REF, *others]:
            q = per[(s, rev)]
            print(f"  REV3 minus {s}: lie {ci(o.lie_auroc - q.lie_auroc, 1, 3)} | error {ci(o.error_auroc - q.error_auroc, 1, 3)}"
                  + (f" | naming {ci(o.naming - q.naming, 100)}" if can_name(ds) else "")
                  + f" | lie_acc {ci(o.acc_policy_lie - q.acc_policy_lie, 100)} | normal "
                  f"{ci(o.acc_policy_normal - q.acc_policy_normal, 100)} | ans80 {ci(o.all_cov80_cert_policy - q.all_cov80_cert_policy, 100)}")
    for s in sysnames:
        q = per[(s, 1)]
        print(f"  {s:30s} lie_AUROC {fmt(q.lie_auroc)}" + (f" | naming {fmt(q.naming, 100, 1)}" if can_name(ds) else ""))


def per_sensor(df, ds):
    """Each sensor's lies vs all benign clips, combined score fitted on validation (as cert_attrib.py)."""
    rows = []
    for fold, g in df[df.available].groupby("fold"):
        va, te = g[g.part == "val"], g[g.part == "test"]
        lv, lt = va.condition.str.startswith("s_swap").to_numpy(), te.condition.str.startswith("s_swap").to_numpy()
        for name, (sig, _) in SYSTEMS.items():
            f = signals(sig)
            Dt = fit_apply(f(va, ds), lv, f(te, ds))[1]
            for sen in CP.EXP[ds]:
                pos = (te.condition == f"s_swap:{sen}").to_numpy()
                keep = pos | ~lt
                rows.append(dict(fold=fold, subject=unit(te), system=name, lie_in=sen,
                                 auroc=auroc(Dt[keep], pos[keep])))
    res = pd.DataFrame(rows)
    print(f"\n{ds}: lie AUROC by lying sensor (rev. 1 score), mean [95% CI] over test subjects")
    w = res.pivot_table(index=["lie_in", "subject"], columns="system", values="auroc")
    for sen, g in w.groupby(level=0, sort=False):
        print(f"  lie in {sen}: " + " | ".join(f"{s} {g[s].mean():.3f}" for s in SYSTEMS)
              + f" | REV3 minus C3 {ci(g[MAIN] - g[REF], 1, 3)}")
    return res


def naming_by_sensor(df, ds):
    """Journal: who gets named when the lie is in each sensor? Rule chosen on the validation person (as
    per_subject does), applied to the test person. One row per test subject, system and lying sensor."""
    from cert_evidential_score import name
    sens = CP.EXP[ds]
    rows = []
    for fold, g in df[df.available].groupby("fold"):
        va, te = g[g.part == "val"], g[g.part == "test"]
        Lv, Lt = va[va.condition.str.startswith("s_swap")], te[te.condition.str.startswith("s_swap")]
        for sysname in (REF, "MI only", MAIN):
            rules = SYSTEMS[sysname][1]
            acc = {q: np.mean(name(Lv, ds, rules, q) == Lv.attacked.to_numpy()) for q in rules}
            rule = max(acc, key=acc.get)
            named, att = name(Lt, ds, rules, rule), Lt.attacked.to_numpy()
            for a in sens:
                k = att == a
                rows.append(dict(fold=fold, subject=unit(te), system=sysname, rule=rule, lie_in=a,
                                 fused_acc=np.mean(Lt.pred.to_numpy()[k] == Lt.y.to_numpy()[k]),
                                 **{f"named_{b}": np.mean(named[k] == b) for b in sens}))
    res = pd.DataFrame(rows)
    print(f"\n{ds}: NAMING by lying sensor (% of that sensor's lie clips; rows = lie in, columns = named), "
          f"mean over test subjects")
    for sysname, g in res.groupby("system", sort=False):
        print(f"  {sysname}")
        for a in sens:
            q = g[g.lie_in == a]
            print(f"    lie in {a:5s}: " + "  ".join(f"{b} {100 * q[f'named_{b}'].mean():5.1f}" for b in sens)
                  + f"   | named correctly {fmt(q[f'named_{a}'], 100, 1)}"
                  + (f" | fused model right on {100 * q.fused_acc.mean():.1f}% of these clips" if sysname == REF else ""))
    return res


def bound_summary():
    files = sorted((ROOT / "results/raw").glob("MI_*.parquet"))
    if not files:
        return None
    r = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    r = r[r.metric.str.startswith("val_mi_")].assign(pair=lambda d: d.metric.str[7:])
    per_fold = r.groupby(["dataset", "version", "pair", "fold"]).value.mean()    # seeds averaged
    out = []
    print("\nDATASET-LEVEL MI lower bound (nats; validation person of each fold; ceiling log 17 = 2.83)")
    for (d, v, p), s in per_fold.groupby(level=[0, 1, 2]):
        n = len(s); m = s.mean(); h = tq(n - 1) * s.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        out.append(dict(dataset=d, version=v, pair=p, n_folds=n, mean=m, lo=m - h, hi=m + h,
                        min_fold=s.min(), max_fold=s.max()))
        print(f"  {d} {v:6s} {p:12s} {m:.2f} [{m - h:.2f}, {m + h:.2f}]  folds {n}, range {s.min():.2f} to {s.max():.2f}")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("d1", "d2", "d1v2", "d2v", "d3"), required=True)
    ap.add_argument("--tag", default="all_s0-1-2", help="journal datasets only: result-file tag "
                    "(all_s0-1-2 = the full run; f0_s0 = the queue C smoke run, for a code check only)")
    args = ap.parse_args(); ds = args.dataset
    journal = ds in CP.JOURNAL
    out_tag = ds
    if journal:
        CP.TAG = args.tag
        journal_files(ds)
        if ds not in CP.BACKUPS:
            raise SystemExit(f"configs/backups.json has no entry for {ds}: run scripts/choose_backups_journal.py first")
        if args.tag != "all_s0-1-2":
            out_tag = f"{ds}_{args.tag}"             # never overwrite the full-run CSVs with a smoke run
    pd.set_option("display.width", 220)

    # ---- main model B4
    ours = add_mi(CP.load(ds), ds)
    per = score_all(ours, ds, CP.backup_answer)
    if journal:
        check_policy(per, ours, ds, f"{ds}, B4")
    else:
        check(per, pd.read_csv(RES / f"e_{ds}_per_subject.csv").query("system == 'ours'"), f"{ds}, B4")
    files = [RES / f"{EVFILE[ds]}{s}.parquet" for s in ("",)]
    ev = pd.concat([pd.read_parquet(f) for f in files if f.exists()], ignore_index=True)
    for mth in MAIN_EV[ds]:
        for rev in (1, 2):
            per[(f"{mth.upper()} alone", rev)] = per_subject(ev[ev.method == mth], ds, ev_feats, EV_RULES, ev_drop, rev)
    table(per, ds, [f"{m.upper()} alone" for m in MAIN_EV[ds]], "certificate on B4")
    out = pd.concat([v.assign(system=k[0], revision=k[1]) for k, v in per.items()]).reset_index()
    out.to_csv(RES / f"mi_{out_tag}_per_subject.csv", index=False)

    # ---- main model TMC (as cert_on_main.py: witnesses judge TMC's answers, drop = TMC without the sensor)
    #      conference datasets: TMC only (as in the paper); journal datasets: TMC and ECML
    for mn in (MAIN_EV[ds] if journal else MAIN_EV[ds][:1]):
        ex = pd.read_parquet(RES / f"{EXPFILE[ds]}_main-{mn}.parquet")
        ex = ex.assign(pred=ex.pred_fused)
        evm = ev[ev.method == mn]
        drops = [c for c in evm.columns if c.startswith("drop_pred_")]
        m = ex.merge(evm[KEYS + ["pred"] + drops].rename(columns={"pred": "pred_ev"}), on=KEYS, how="inner",
                     validate="one_to_one")
        if len(m) != len(ex) or (m.pred == m.pred_ev).mean() < 0.999:
            raise SystemExit(f"{mn}: expert and evidential rows do not match: stop and paste this")
        m = add_mi(m, ds)
        pt = score_all(m, ds, ev_drop)
        if not journal:
            stored = pd.read_csv(RES / f"e_{ds}_on_{mn}_per_subject.csv")
            stored = stored[stored.system == f"certificate on {mn}"]
            check(pt, stored, f"{ds}, on {mn}")
        for rev in (1, 2):
            pt[(f"{mn.upper()} alone", rev)] = per[(f"{mn.upper()} alone", rev)]
        table(pt, ds, [f"{mn.upper()} alone"], f"certificate on {mn.upper()}")
        out = pd.concat([v.assign(system=k[0], revision=k[1]) for k, v in pt.items()]).reset_index()
        out.to_csv(RES / f"mi_{out_tag}_on_{mn}_per_subject.csv", index=False)

    if ds == "d1" or journal:
        per_sensor(ours, ds).to_csv(RES / f"mi_{out_tag}_per_sensor.csv", index=False)
    if journal:
        naming_by_sensor(ours, ds).to_csv(RES / f"mi_{out_tag}_naming_by_sensor.csv", index=False)
    b = bound_summary()
    if b is not None:
        b.to_csv(RES / "mi_bound_summary.csv", index=False)
    print(f"\nsaved: results/certificate/mi_{out_tag}_*.csv" + (", mi_bound_summary.csv" if b is not None else ""))


if __name__ == "__main__":
    main()
