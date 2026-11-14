"""Addition 2 (docs/plan_additions_2026-11-14.md): partial IMU-off on WearGait-PD. Score every system on clips in
which k IMUs of one body group are switched off (training mean) during p = 25 / 50 / 75 % of the clip.

  python scripts/cert_imuoff2_score.py                 full run (queue K files, tag all_s0-1-2)
  python scripts/cert_imuoff2_score.py --tag f0_s0     smoke-run files (code check only; never overwrites)

Exactly the scoring of the whole-clip IMU-off test (scripts/cert_imuoff_score.py, imported, not edited), done once
per p: FROZEN systems fitted on each fold's validation people with the lie bench only; unit = fold, seeds pooled
within a fold; 95% intervals over the 10 folds. Revision 1 is the main result.
MAIN RESULT, named in advance (plan 2.3): legs, REV3 (main) use / drop rev. 1 minus doing nothing, at each k and p.
PREDICTIONS I1-I4 (plan 2.4) are graded against the STORED whole-clip results (imuoff_d3_per_fold.csv, p = 100%).
CHECKS before any number: (1) clean rows identical to the lie bench; (2) each frozen fit reproduces the stored lie
numbers exactly; (3) the bench's clean rows identical to the stored whole-clip IMU-off bench's clean rows.
Writes results/certificate/imuoff2_d3[_<tag>]_per_fold.csv, _answers_per_fold.csv, _predictions.csv
"""
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import cert_policy as CP
import cert_mi_score as MS
import cert_dropout_score as DSC
import cert_imuoff_score as IS
from cert_evidential_score import per_subject, ev_feats, ev_drop, EV_RULES, RES, EVFILE
from cert_score import unit
from cert_fault_score import (Frozen, check_clean, check_frozen_on_lies, with_dropout, merge, FULL, KEYS,
                              SYS_CONF, SYS_C3, SYS_MAIN, SYS_D, REV3_SIG, REV3_RULES)
from cert_conditions import IMUS, IMUOFF2_P
from bench_subset import check_subset

DS_ = "d3"
SENS = tuple(IMUS)


def bench(tag):
    ex = pd.read_parquet(RES / f"experts_{DS_}_{tag}_imuoff2.parquet")
    ex = ex.assign(pred=ex.pred_fused)
    mi = pd.read_parquet(RES / f"mi_{DS_}_own_{tag}_imuoff2.parquet")
    b = merge(ex, mi, [c for c in mi.columns if c.startswith(("pmi_", "mi_", "s4"))], f"{DS_} partial IMU-off MI")
    if not b.available.all() or not (b.part == "test").all():
        raise SystemExit("partial IMU-off bench: unexpected part or unavailable rows: stop and paste this")
    return b


def p_of(d):
    """p per row (0 for clean)."""
    return d.condition.astype(str).str.extract(r":p(\d+)$")[0].fillna(0).astype(int).to_numpy()


def as_whole(d):
    """io_part:<g>:k<k>:p<p> -> io_fill:<g>:k<k>, so the whole-clip scorer's functions read it unchanged."""
    d = d.copy()
    d["condition"] = d.condition.astype(str).str.replace(r"^io_part:(.*):p\d+$", r"io_fill:\1", regex=True)
    return d


def check_shared(tag):
    """The clean rows of this bench must equal the clean rows of the stored whole-clip IMU-off bench."""
    for pre, by in ((f"experts_{DS_}", ()), (f"mi_{DS_}_own", ()), (f"evidential_{DS_}", ("method",)),
                    (f"dropout_{DS_}_sd50", ())):
        old = pd.read_parquet(RES / f"{pre}_{FULL}_imuoff.parquet")
        new = pd.read_parquet(RES / f"{pre}_{tag}_imuoff2.parquet")
        check_subset(old[old.condition == "clean"], new, by, f"{pre} imuoff2 vs imuoff")


def gains(pf, p):
    """Per (group, k): per-fold REV3 rev. 1 gain over doing nothing in points, folds as columns."""
    m = pf[(pf.system == SYS_MAIN) & (pf.revision == 1) & (pf.group != "clean") & (pf.p == p)].copy()
    m["diff"] = 100 * (m.acc_policy - m.acc_alone)
    return m.pivot_table(index=["group", "k"], columns="fold", values="diff")


def predictions(pf):
    old = pd.read_csv(RES / f"imuoff_{DS_}_per_fold.csv").assign(p=100)
    v = lambda ok: "CONFIRMED" if ok else "NOT CONFIRMED"
    out = []
    print("\nMAIN RESULT (named in advance): legs, REV3 (main) use / drop rev. 1 minus doing nothing, points,"
          " mean over folds [95% interval], folds better / worse; p = 100 is the stored whole-clip test")
    g = {p: gains(pf if p != 100 else old, p) for p in (*IMUOFF2_P, 100)}
    for p in (*IMUOFF2_P, 100):
        print(f"  p = {p}%")
        for kk, row in g[p].loc["legs"].iterrows():
            mm, lo, hi = IS.mci(row.to_numpy())
            print(f"    k = {kk}: {mm:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]  better {int((row > 0).sum())}, "
                  f"worse {int((row < 0).sum())} of {row.notna().sum()}")
            if p != 100:
                out.append(dict(item="main", group="legs", k=kk, p=p, mean=mm, lo=lo, hi=hi,
                                n_better=int((row > 0).sum()), n_worse=int((row < 0).sum()), n_folds=int(row.notna().sum())))
    g8 = {p: g[p].loc[("legs", 8)].mean() for p in g}
    i1 = g8[25] < g8[50] < g8[75] < g8[100]
    rv = lambda d, p: d[(d.system == SYS_MAIN) & (d.revision == 1) & (d.group == "legs") & (d.k == 8) & (d.p == p)]
    nothing25, nothing100 = 100 * rv(pf, 25).acc_alone.mean(), 100 * rv(old, 100).acc_alone.mean()
    i2 = nothing25 >= nothing100 + 20
    au = {p: rv(pf if p != 100 else old, p).auroc.mean() for p in (*IMUOFF2_P, 100)}
    i3 = au[25] < au[50] < au[75]
    k1 = {(s, p): g[p].loc[(s, 1)].mean() for s in SENS for p in IMUOFF2_P}
    i4 = all(val < 0.25 for val in k1.values())
    items = [("I1", i1, "legs k=8 gain: " + ", ".join(f"p{p} {g8[p]:+.2f}" for p in g8)),
             ("I2", i2, f"legs k=8 doing nothing: p25 {nothing25:.1f}% vs whole clip {nothing100:.1f}%"),
             ("I3", i3, "legs k=8 REV3 AUROC: " + ", ".join(f"p{p} {au[p]:.3f}" for p in au)),
             ("I4", i4, "k=1 gain: " + ", ".join(f"{s} p{p} {val:+.2f}" for (s, p), val in k1.items()))]
    print("\nPREDICTIONS (plan, Addition 2.4), each with a plain verdict:")
    for n, ok, val in items:
        print(f"  {n}: {val} -> {v(ok)}")
        out.append(dict(item=n, verdict=v(ok), value=val))
    for s in ("arms", "trunk"):
        for p in IMUOFF2_P:
            for kk, row in g[p].loc[s].iterrows():
                mm, lo, hi = IS.mci(row.to_numpy())
                out.append(dict(item="supporting", group=s, k=kk, p=p, mean=mm, lo=lo, hi=hi,
                                n_better=int((row > 0).sum()), n_worse=int((row < 0).sum()), n_folds=int(row.notna().sum())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=FULL, help=f"{FULL} = full run; e.g. f0_s0 = smoke-run files (code check)")
    args = ap.parse_args(); ds = DS_
    MS.journal_files(ds)
    check_shared(args.tag)
    b = bench(args.tag)
    folds, seeds = sorted(b.fold.unique()), sorted(b.seed.unique())
    keep = lambda d: d[d.fold.isin(folds) & d.seed.isin(seeds)]
    lie = keep(MS.add_mi(CP.load(ds), ds)); lie = lie[lie.available]
    check_clean(lie, b, ["pred", "msp", "exp_js", "s4"] + [f"exp_support_{m}" for m in CP.EXP[ds]], f"{ds} B4")
    ev_l = keep(pd.read_parquet(RES / f"{EVFILE[ds]}.parquet")); ev_l = ev_l[ev_l.available]
    ev_b = pd.read_parquet(RES / f"evidential_{ds}_{args.tag}_imuoff2.parquet")
    for m in MS.MAIN_EV[ds]:
        check_clean(ev_l[ev_l.method == m], ev_b[ev_b.method == m], ["pred", "msp", "u", "conflict"], f"{ds} {m}")
    d_l = with_dropout(lie, RES / f"dropout_{ds}_sd50_{FULL}.parquet", ds)
    raw = b.copy()
    d_b = with_dropout(b, RES / f"dropout_{ds}_sd50_{args.tag}_imuoff2.parquet", ds)
    check_clean(d_l, d_b, ["pred", "msp"], f"{ds} dropout sd50")
    raw = raw.merge(d_b[KEYS + ["sd50_pred"] + [f"sd50_hide_pred_{g}" for g in SENS]], on=KEYS, validate="one_to_one")

    rev3 = MS.signals(REV3_SIG)
    systems = {
        SYS_CONF: (lie, b, MS.signals(["s1"]), None, None),
        SYS_C3: (lie, b, MS.signals(["s1", "s2", "s3"]), {"odd one out": MS.ODD}, CP.backup_answer),
        SYS_MAIN: (lie, b, rev3, REV3_RULES, CP.backup_answer),
        **{f"{m.upper()} alone": (ev_l[ev_l.method == m], ev_b[ev_b.method == m], ev_feats, EV_RULES, ev_drop)
           for m in MS.MAIN_EV[ds]},
        SYS_D: (d_l, d_b, rev3, REV3_RULES, DSC.hide_answer("sd50_hide_pred")),
    }
    rows = []
    for name_, (L, B, feats, rules, drop) in systems.items():
        for rev in (1, 2):
            ref = per_subject(L, ds, feats, rules or {}, drop or (lambda d, ds_, n: d.pred.to_numpy()), rev) \
                if rules else None
            for fold in folds:
                va, te_l = L[(L.fold == fold) & (L.part == "val")], L[(L.fold == fold) & (L.part == "test")]
                te = B[(B.fold == fold).to_numpy()]
                fz = Frozen(va, ds, feats, rules, drop, rev)
                if ref is not None:
                    check_frozen_on_lies(fz, te_l, ref.loc[unit(te_l)], f"{ds} {name_} rev {rev} fold {fold}")
                D, named, final, _ = fz.apply(te)
                pp = p_of(te)
                for p in IMUOFF2_P:
                    k = (pp == 0) | (pp == p)                         # clean clips + this p
                    rows += [dict(r, p=p) for r in IS.fold_metrics(as_whole(te[k]), D[k], named[k], final[k],
                                                                    fz.t, name_, rev)]
        print(f"scored: {name_}" + (" (frozen fit reproduces the lie numbers exactly)" if rules else ""))
    pf = pd.DataFrame(rows)
    n_rows = pf.groupby(["system", "revision"]).n.sum()
    if n_rows.nunique() != 1:
        raise SystemExit(f"systems scored on different numbers of rows: {n_rows.to_dict()}: stop and paste this")
    n_cond = len(pf[pf.k > 0][["group", "k", "p"]].drop_duplicates())
    if n_cond != sum(IMUS.values()) * len(IMUOFF2_P):
        raise SystemExit(f"{n_cond} partial IMU-off conditions scored, expected 39: stop and paste this")
    print(f"check: every system scored on the same {int(n_rows.iloc[0])} rows ({len(folds)} folds, {len(seeds)} seeds,"
          f" {n_cond} partial IMU-off conditions; clean rows counted once per p)")

    pr_ = p_of(raw)
    ans = pd.concat([IS.answers(as_whole(raw[(pr_ == 0) | (pr_ == p)])).assign(p=p) for p in IMUOFF2_P],
                    ignore_index=True)
    for p in IMUOFF2_P:
        print(f"\n======== p = {p}% of the clip ========")
        IS.report(pf[pf.p == p].drop(columns="p"), ans[ans.p == p].drop(columns="p"))
    pr = predictions(pf) if len(folds) > 2 else None
    if pr is None:
        print("\nsmoke run: fewer than 3 folds, predictions not evaluated (code check only)")
    sfx = "" if args.tag == FULL else f"_{args.tag}"
    pf.to_csv(RES / f"imuoff2_{ds}{sfx}_per_fold.csv", index=False)
    ans.to_csv(RES / f"imuoff2_{ds}{sfx}_answers_per_fold.csv", index=False)
    if pr is not None:
        pr.to_csv(RES / f"imuoff2_{ds}{sfx}_predictions.csv", index=False)
    print(f"\nsaved: results/certificate/imuoff2_{ds}{sfx}_per_fold.csv, _answers_per_fold.csv"
          + (", _predictions.csv" if pr is not None else ""))


if __name__ == "__main__":
    main()
