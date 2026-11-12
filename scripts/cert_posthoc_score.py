"""Follow-up plan, Part B (docs/plan_followup_2026-11-12.md): two rule changes to REV3, POST HOC (designed after
seeing results). Scored next to REV3 revision 1, never instead of it; every printed table says "post hoc".

  python scripts/cert_posthoc_score.py                          every bench
  python scripts/cert_posthoc_score.py --benches lies,faults    some benches only

Rule (a), confidence as a second alarm: alarm = max(percentile of REV3's score D, percentile of 1 - msp), both
           percentiles (mid-rank) among the fold's VALIDATION clean clips. Used for detection (AUROC) and for
           abstaining ("answer the safest 80%"). The use / drop decision stays revision 1's.
Rule (b), drop only when sure which sensor: naming margin = for "odd one out" the named sensor's divergence minus
           the second largest; for "lowest MI share" the second lowest share minus the lowest. Drop when D >= t
           AND margin >= tau, else keep the main model's answer. t = revision 1's threshold. tau chosen on the
           fold's VALIDATION lie bench from {0} and the 5..95% quantiles (step 5%) of the margin among validation
           clips with D >= t, highest validation accuracy, ties -> the smallest tau.
Variants: REV3 (main, revision 1) | + (a) | + (b) | + (a) + (b); plus "confidence (1 - msp)" as a reference score.
Fitted once per fold on the validation LIE bench (cert_fault_score.Frozen + the two additions) and applied
unchanged to every bench of that fold: lies (d1v2, d2v, d3), faults (d1v2, d2v, d3), no-watch (d2v), real dropouts
(d3), IMU-off (d3). Metrics come from each bench's own scorer functions (imported, not edited).
CHECK FIRST, per bench: the REV3 (main) variant computed here reproduces that bench's stored REV3 revision 1
numbers exactly (difference <= 1e-9), so the post hoc rows differ from the main row only by the rule.
Writes results/certificate/posthoc_per_unit.csv, posthoc_imuoff_per_fold.csv and posthoc_tau.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import cert_policy as CP
import cert_mi_score as MS
import cert_fault_score as FS
import cert_nowatch_score as NW
import cert_realdrop_score as RD
from cert_evidential_score import name, is_lie, ci, RES
from cert_score import auroc, unit

FULL = FS.FULL
V0, VA, VB, VAB, VC = ("REV3 (main)", "REV3 + (a) confidence alarm [post hoc]", "REV3 + (b) sure-naming gate [post hoc]",
                       "REV3 + (a) + (b) [post hoc]", "confidence (1 - msp) [reference]")
VARIANTS = (V0, VA, VB, VAB, VC)
TOL = 1e-9
BENCHES = ("lies", "faults", "nowatch", "realdrop", "imuoff")
DATASETS = {"lies": ("d1v2", "d2v", "d3"), "faults": ("d1v2", "d2v", "d3"), "nowatch": ("d2v",),
            "realdrop": ("d3",), "imuoff": ("d3",)}
QS = np.round(np.arange(0.05, 0.951, 0.05), 2)


# ---------------------------------------------------------------- the two rules
def margin(d, ds, rule):
    col, how = rule
    s = np.sort(d[[f"{col}_{m}" for m in CP.EXP[ds]]].to_numpy(float), axis=1)
    return s[:, -1] - s[:, -2] if how == "max" else s[:, 1] - s[:, 0]


def pct(ref_sorted, v):
    """Mid-rank percentile of v among the reference values (ties count half)."""
    lo = np.searchsorted(ref_sorted, v, "left"); hi = np.searchsorted(ref_sorted, v, "right")
    return (lo + 0.5 * (hi - lo)) / len(ref_sorted)


class FrozenPH(FS.Frozen):
    """REV3 revision 1 exactly as cert_fault_score.Frozen, plus the alarm calibration (a) and the gate tau (b)."""

    def __init__(self, va, ds):
        sig, rules = MS.SYSTEMS[MS.MAIN]
        super().__init__(va, ds, MS.signals(sig), rules, CP.backup_answer, 1)
        Dv = self._lin(self.feats(va, ds), self.w)
        cv = (va.condition == "clean").to_numpy()
        self.cal_D, self.cal_c = np.sort(Dv[cv]), np.sort(1 - va.msp.to_numpy()[cv])
        yv, fv = va.y.to_numpy(), va.pred.to_numpy()
        bv = self.drop(va, ds, name(va, ds, self.rules, self.rule))
        mv, fl = margin(va, ds, self.rules[self.rule]), Dv >= self.t
        self.tau, self.tau_q = 0.0, 0.0
        if fl.any():
            cand = np.r_[0.0, np.quantile(mv[fl], QS)]
            acc = [np.mean(np.where(fl & (mv >= c), bv, fv) == yv) for c in cand]
            i = int(np.argmax(acc))                        # first maximum = the smallest tau among ties
            self.tau, self.tau_q = float(cand[i]), float(np.r_[0.0, QS][i])
        self.n_val_flagged = int(fl.sum())

    def outputs(self, te):
        """{variant: (score, final answer, abstain score)}, plus named, dropped (rev 1) and dropped (gate)."""
        D, named, final1, A = self.apply(te)
        fl = D >= self.t
        gate = fl & (margin(te, self.ds, self.rules[self.rule]) >= self.tau)
        final_b = np.where(gate, self.drop(te, self.ds, named), te.pred.to_numpy())
        c = 1 - te.msp.to_numpy()
        alarm = np.maximum(pct(self.cal_D, D), pct(self.cal_c, c))
        out = {V0: (D, final1, A), VA: (alarm, final1, alarm), VB: (D, final_b, D), VAB: (alarm, final_b, alarm),
               VC: (c, te.pred.to_numpy(), c)}
        return out, named, fl, gate


# ---------------------------------------------------------------- per-bench metrics (each bench's own functions)
def lie_metrics(te, sc, final, A, named):
    lt, yt, grp = is_lie(te), te.y.to_numpy(), CP.group(te.condition)
    return dict(lie_auroc=auroc(sc, lt), naming=np.mean(named[lt] == te.attacked.to_numpy()[lt]),
                acc_policy_lie=np.mean(final[grp == "lie"] == yt[grp == "lie"]),
                acc_policy_normal=np.mean(final[grp == "normal"] == yt[grp == "normal"]),
                all_cov80_cert_policy=CP.selective(A, final, yt, 0.8))


def fault_metrics(te, sc, final, A, named):
    r = [x for x in FS.fault_rows(te, sc, named, final, A, np.inf, "", 1) if x["group"] == "all faults"][0]
    return {k: r.get(k, np.nan) for k in ("auroc", "auroc_harmful", "acc_fault", "acc_policy_fault",
                                          "acc_policy_clean", "cov80_policy", "naming_harmful")}


# ---------------------------------------------------------------- loading
def lie_bench(ds):
    MS.journal_files(ds)
    lie = MS.add_mi(CP.load(ds), ds)
    return lie[lie.available]


def check(got, stored, cols, label):
    worst = 0.0
    for c in cols:
        x, y = np.asarray(got[c], float), np.asarray(stored[c], float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            raise SystemExit(f"{label}: NaN pattern of {c} differs from the stored numbers: stop and paste this")
        if (~np.isnan(x)).any():
            worst = max(worst, float(np.nanmax(np.abs(x - y))))
    if worst > TOL:
        raise SystemExit(f"{label}: REV3 (main) differs from the stored numbers by {worst:.2e}: stop and paste this")
    print(f"check ({label}): REV3 (main) reproduces the stored revision-1 numbers exactly (largest diff {worst:.1e})")


# ---------------------------------------------------------------- benches
def run_lies_faults(ds, benches, taus, rows):
    lie = lie_bench(ds)
    flt = FS.fault_frame(ds, FULL, "_faults") if "faults" in benches else None
    if flt is not None:
        FS.check_clean(lie, flt, ["pred", "msp", "exp_js", "s4"] + [f"exp_support_{m}" for m in CP.EXP[ds]], f"{ds} B4")
    for fold in sorted(lie.fold.unique()):
        va, te = lie[(lie.fold == fold) & (lie.part == "val")], lie[(lie.fold == fold) & (lie.part == "test")]
        fz = FrozenPH(va, ds)
        taus.append(dict(dataset=ds, fold=fold, unit=unit(te), rule=fz.rule, t=fz.t, tau=fz.tau, tau_quantile=fz.tau_q,
                         n_val_flagged=fz.n_val_flagged))
        for bench, T, f in (("lies", te, lie_metrics),
                            ("faults", None if flt is None else flt[(flt.fold == fold) & (flt.part == "test")],
                             fault_metrics)):
            if bench not in benches:
                continue
            out, named, fl, gate = fz.outputs(T)
            for v, (sc, final, A) in out.items():
                rows.append(dict(bench=bench, dataset=ds, fold=fold, unit=unit(T), variant=v,
                                 dropped=np.mean(fl if v in (V0, VA) else gate if v != VC else 0 * fl),
                                 **f(T, sc, final, A, named)))
    for bench, f, cols, sel in (("lies", f"mi_{ds}_per_subject.csv", ["lie_auroc", "naming", "acc_policy_lie",
                                 "acc_policy_normal", "all_cov80_cert_policy"], lambda s: s[(s.system == MS.MAIN)]),
                                ("faults", f"faults_{ds}_per_subject.csv", ["auroc", "auroc_harmful", "acc_policy_fault",
                                 "acc_policy_clean", "cov80_policy", "naming_harmful"],
                                 lambda s: s[(s.system == FS.SYS_MAIN) & (s.group == "all faults")])):
        if bench not in benches:
            continue
        s = sel(pd.read_csv(RES / f)); s = s[s.revision == 1].assign(unit=lambda d: d.subject.astype(str))
        g = pd.DataFrame([r for r in rows if r["bench"] == bench and r["dataset"] == ds and r["variant"] == V0])
        g = g.assign(unit=g.unit.astype(str)).set_index("unit")
        check(g, s.set_index("unit").loc[g.index], cols, f"{ds} {bench}")


def run_nowatch(rows):
    ds = "d2v"
    lie = lie_bench(ds)
    b = NW.bench(FULL)
    FS.check_clean(lie, b[b.part == "test"], ["pred", "msp", "exp_js", "s4"], f"{ds} no-watch B4")
    sets, notes = NW.reference_sets(b)
    per = []
    for fold in sorted(b.fold.unique()):
        va = lie[(lie.fold == fold) & (lie.part == "val")]
        kb = (b.fold == fold).to_numpy()
        fz = FrozenPH(va, ds)
        out, named, fl, gate = fz.outputs(b[kb])
        for v, (sc, final, A) in out.items():
            R = NW.row_frame(b[kb], sc, named, final, A, np.inf, v, 1, sets[kb])
            for p, g in R.groupby("subject"):
                per.append(dict(variant=v, subject=p, fold=fold, **NW.person_metrics(g)))
    per = pd.DataFrame(per)
    st = pd.read_csv(RES / "nowatch_d2v_per_subject.csv")
    st = st[(st.system == FS.SYS_MAIN) & (st.revision == 1)].set_index("subject")
    g = per[per.variant == V0].set_index("subject")
    check(g, st.loc[g.index], ["correct_policy_nw", "correct_policy_ref", "auroc_nw", "cov80_policy_nw"], "d2v no-watch")
    seeds = b.seed.nunique()
    n_clips = g.n_nw // seeds
    big = sorted(n_clips[n_clips >= NW.MIN_BIG].index)
    for v, q in per.groupby("variant", sort=False):
        q = q.set_index("subject")
        rows.append(dict(bench="nowatch", dataset=ds, unit="pooled / mean of large people", variant=v,
                         acc_policy_nw=q.correct_policy_nw.sum() / q.n_nw.sum(),
                         acc_policy_ref=q.correct_policy_ref.sum() / q.n_ref.sum(),
                         auroc_nw_mean4=q.loc[big].auroc_nw.mean(), cov80_nw_mean4=q.loc[big].cov80_policy_nw.mean(),
                         acc_policy_nw_mean4=(q.loc[big].correct_policy_nw / q.loc[big].n_nw).mean(),
                         acc_policy_nw_person2=q.loc[2].correct_policy_nw / q.loc[2].n_nw if 2 in q.index else np.nan))


def run_realdrop(rows):
    ds = "d3"
    lie = lie_bench(ds)
    b = RD.bench(FULL)
    FS.check_clean(lie, b[b.part == "test"], ["pred", "msp", "exp_js", "s4"], f"{ds} real-dropout B4")
    sets = RD.sets_of(b)
    R = []
    for fold in sorted(b.fold.unique()):
        va = lie[(lie.fold == fold) & (lie.part == "val")]
        kb = (b.fold == fold).to_numpy()
        out, named, fl, gate = FrozenPH(va, ds).outputs(b[kb])
        for v, (sc, final, A) in out.items():
            R.append(RD.rows_for(b[kb], sc, named, final, A, np.inf, v, 1, sets[kb]))
    R = pd.concat(R, ignore_index=True)
    pf = RD.per_fold(R)
    st = pd.read_csv(RES / "realdrop_d3_per_fold.csv")
    st = st[(st.system == FS.SYS_MAIN) & (st.revision == 1)].set_index("fold")
    g = pf[pf.system == V0].set_index("fold")
    check(g, st.loc[g.index], ["auroc", "acc_alone_rd", "acc_policy_rd", "acc_policy_ref"], "d3 real dropouts")
    for (v, f), q in R.groupby(["system", "fold"], sort=False):
        both = q[q.set.isin(["rd1", "ref"])]
        p = pf[(pf.system == v) & (pf.fold == f)].iloc[0]
        rows.append(dict(bench="realdrop", dataset=ds, fold=f, unit=f"fold{f:02d}", variant=v,
                         n_rd1=int((q.set == "rd1").sum()), correct_policy_rd1=int(((q.final == q.y) & (q.set == "rd1")).sum()),
                         acc_policy_rd1=p.acc_policy_rd, acc_alone_rd1=p.acc_alone_rd, acc_policy_ref=p.acc_policy_ref,
                         auroc_rd1=p.auroc, cov80_rd1_ref=CP.selective(both.abstain_score.to_numpy(),
                                                                         both.final.to_numpy(), both.y.to_numpy(), 0.8)))


def run_imuoff():
    import cert_imuoff_score as IO
    ds = "d3"
    lie = lie_bench(ds)
    b = IO.bench(FULL)
    FS.check_clean(lie, b, ["pred", "msp", "exp_js", "s4"], f"{ds} IMU-off B4")
    out_rows = []
    for fold in sorted(b.fold.unique()):
        va = lie[(lie.fold == fold) & (lie.part == "val")]
        te = b[(b.fold == fold).to_numpy()]
        out, named, fl, gate = FrozenPH(va, ds).outputs(te)
        for v, (sc, final, A) in out.items():
            out_rows += IO.fold_metrics(te, sc, named, final, np.inf, v, 1)
    pf = pd.DataFrame(out_rows)
    st = pd.read_csv(RES / "imuoff_d3_per_fold.csv")
    st = st[(st.system == FS.SYS_MAIN) & (st.revision == 1)].set_index(["fold", "group", "k"])
    g = pf[pf.system == V0].set_index(["fold", "group", "k"])
    check(g, st.loc[g.index], ["acc_alone", "acc_policy", "auroc"], "d3 IMU-off")
    return pf


# ---------------------------------------------------------------- report
def report(res, taus, io):
    pd.set_option("display.width", 230)
    print("\n" + "=" * 100 + "\nPOST HOC RULES (designed after seeing results; labelled; never the main result)\n" + "=" * 100)
    t = pd.DataFrame(taus)
    print("\nRule (b): tau chosen on the validation lie bench, per dataset: folds with tau = 0 (rule = revision 1), "
          "median quantile of the chosen tau")
    print(pd.DataFrame({"folds": t.groupby("dataset", sort=False).size(),
                        "tau_zero": (t.tau == 0).groupby(t.dataset, sort=False).sum(),
                        "median_quantile": t.groupby("dataset", sort=False).tau_quantile.median()}).to_string())
    specs = {"lies": ["lie_auroc", "acc_policy_lie", "acc_policy_normal", "all_cov80_cert_policy", "dropped"],
             "faults": ["auroc_harmful", "auroc", "acc_policy_fault", "acc_policy_clean", "cov80_policy", "dropped"],
             "realdrop": ["auroc_rd1", "acc_policy_rd1", "acc_policy_ref", "cov80_rd1_ref"]}
    for bench, cols in specs.items():
        q = res[res.bench == bench]
        for ds, g in q.groupby("dataset", sort=False):
            print(f"\n{bench.upper()} {ds} (mean over {g.unit.nunique()} units; accuracy as a share)")
            print(g.groupby("variant", sort=False)[cols].mean().round(3).to_string())
            a = g[g.variant == V0].set_index("unit")
            for v in (VA, VB, VAB):
                b = g[g.variant == v].set_index("unit").loc[a.index]
                print(f"  {v} minus REV3: " + " | ".join(f"{c} {ci(b[c] - a[c], 100 if 'auroc' not in c else 1, 1 if 'auroc' not in c else 3)}"
                                                       for c in cols if c != "dropped"))
            if bench == "realdrop":
                sums = g.groupby("variant", sort=False)[["correct_policy_rd1", "n_rd1"]].sum()
                pooled = sums.correct_policy_rd1 / sums.n_rd1
                print("  pooled accuracy on single-group real-dropout clips: "
                      + " | ".join(f"{v}: {100 * x:.1f}" for v, x in pooled.items()))
    q = res[res.bench == "nowatch"]
    if len(q):
        print("\nNO-WATCH d2v (pooled accuracy over all clips; AUROC, 80% and accuracy also as the mean of the 4 large people)")
        print(q.set_index("variant")[["acc_policy_nw", "acc_policy_nw_mean4", "acc_policy_nw_person2", "acc_policy_ref",
                                      "auroc_nw_mean4", "cov80_nw_mean4"]].round(3).to_string())
    if io is not None:
        m = io[io.k > 0].assign(diff=lambda d: 100 * (d.acc_policy - d.acc_alone))
        print("\nIMU-OFF d3: use / drop minus doing nothing, points, mean over folds (REV3 and the gate (b))")
        print(m[m.system.isin([V0, VB])].pivot_table(index=["group", "k"], columns="system", values="diff", sort=False)
              .round(2).to_string())
        print("\nIMU-OFF d3: detection AUROC (IMU-off vs clean), mean over folds")
        print(m[m.system.isin([V0, VA, VC])].pivot_table(index=["group", "k"], columns="system", values="auroc", sort=False)
              .round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default=",".join(BENCHES))
    benches = [b for b in ap.parse_args().benches.split(",") if b]
    bad = set(benches) - set(BENCHES)
    if bad:
        raise SystemExit(f"unknown benches {bad}; choose from {BENCHES}")
    rows, taus, io = [], [], None
    for ds in ("d1v2", "d2v", "d3"):
        if {"lies", "faults"} & set(benches):
            run_lies_faults(ds, [b for b in benches if b in ("lies", "faults")], taus, rows)
    if "nowatch" in benches:
        run_nowatch(rows)
    if "realdrop" in benches:
        run_realdrop(rows)
    if "imuoff" in benches:
        io = run_imuoff()
    res = pd.DataFrame(rows)
    report(res, taus, io)
    res.to_csv(RES / "posthoc_per_unit.csv", index=False)
    pd.DataFrame(taus).to_csv(RES / "posthoc_tau.csv", index=False)
    if io is not None:
        io.to_csv(RES / "posthoc_imuoff_per_fold.csv", index=False)
    print("\nsaved: results/certificate/posthoc_per_unit.csv, posthoc_tau.csv"
          + (", posthoc_imuoff_per_fold.csv" if io is not None else ""))


if __name__ == "__main__":
    main()
