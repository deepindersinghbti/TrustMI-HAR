"""IMU-off test (docs/plan_followup_2026-11-12.md, Part A): score every system on WearGait-PD clips in which k IMUs
of one body group are switched off (set to the training mean for the whole clip).

  python scripts/cert_imuoff_score.py                  full run (queue J files, tag all_s0-1-2)
  python scripts/cert_imuoff_score.py --tag f0_s0      smoke-run files (code check only; never overwrites)

FROZEN, as for the faults, the no-watch clips and the real dropouts: every system is fitted on each fold's
VALIDATION people with the LIE bench only (cert_fault_score.Frozen) and applied unchanged to the fold's test people:
  clean                  the clean test clips
  io_fill:<group>:k<k>   the same clips with the first k IMUs of the clip's random order in that group switched off
Answers without a certificate: doing nothing (main model); known missing (the backup for the WHOLE group,
configs/backups.json); M (the group flagged absent in the main model); the dropout model with the group hidden.
MAIN RESULT, named in advance (plan A.5): legs, REV3 (main) use / drop revision 1 minus doing nothing, at each
k = 1..8, mean over the 10 folds with a 95% interval over folds (fold = unit, seeds pooled within a fold).
PREDICTIONS P1-P5 (plan A.6) are evaluated here and each gets a plain verdict.
CHECKS before any number: (1) clean rows identical to the lie bench; (2) each frozen fit reproduces the stored lie
numbers exactly; (3) the k = all rows equal the r_fill:<group> rows of the real-dropout bench, column by column.
Writes results/certificate/imuoff_d3[_<tag>]_per_fold.csv, _answers_per_fold.csv, _predictions.csv and
_rev3_rows.parquet
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import cert_policy as CP
import cert_mi_score as MS
import cert_dropout_score as DSC
from cert_evidential_score import per_subject, ev_feats, ev_drop, EV_RULES, RES, EVFILE
from cert_score import auroc, unit, tq
from cert_fault_score import (Frozen, check_clean, check_frozen_on_lies, with_dropout, merge, FULL, KEYS,
                              SYS_CONF, SYS_C3, SYS_MAIN, SYS_D, REV3_SIG, REV3_RULES)
from cert_conditions import IMUS

DS_ = "d3"
SENS = tuple(IMUS)                                       # arms, trunk, legs
TOL = 1e-6                                               # as check_clean: same inputs give the same outputs


# ---------------------------------------------------------------- data
def bench(tag):
    ex = pd.read_parquet(RES / f"experts_{DS_}_{tag}_imuoff.parquet")
    ex = ex.assign(pred=ex.pred_fused)
    mi = pd.read_parquet(RES / f"mi_{DS_}_own_{tag}_imuoff.parquet")
    b = merge(ex, mi, [c for c in mi.columns if c.startswith(("pmi_", "mi_", "s4"))], f"{DS_} IMU-off MI")
    if not b.available.all() or not (b.part == "test").all():
        raise SystemExit("IMU-off bench: unexpected part or unavailable rows: stop and paste this")
    return b


def group_k(d):
    """(group, k) per row; ('', 0) for clean."""
    c = d.condition.astype(str)
    k = c.str.extract(r":k(\d+)$")[0].fillna(0).astype(int).to_numpy()
    g = np.where(c.to_numpy() == "clean", "", d.attacked.astype(str).to_numpy())
    return g, k


def mci(v):
    """mean, lower, upper of a 95% t-interval (as everywhere in the project)."""
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v)
    m = v.mean() if n else np.nan
    h = tq(n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - h, m + h


# ---------------------------------------------------------------- check 3: k = all equals r_fill
def compare(old, new, keys, label):
    cols = sorted((set(old.columns) & set(new.columns)) - set(keys))
    m = old.merge(new, on=keys, how="inner", suffixes=("_o", "_n"), validate="one_to_one")
    if len(m) != len(new) or len(new) == 0:
        raise SystemExit(f"{label}: {len(new)} k=all rows, {len(m)} matched with r_fill: stop and paste this")
    worst, bad = 0.0, []
    for c in cols:
        x, y = m[f"{c}_o"].to_numpy(), m[f"{c}_n"].to_numpy()
        if np.issubdtype(x.dtype, np.floating) and np.issubdtype(y.dtype, np.floating):
            d = float(np.nanmax(np.abs(x - y))) if len(x) else 0.0
            worst = max(worst, d)
            if d > TOL or not np.array_equal(np.isnan(x), np.isnan(y)):
                bad.append(f"{c} ({d:.1e})")
        elif not np.array_equal(x.astype(str), y.astype(str)):
            bad.append(f"{c} ({int((x.astype(str) != y.astype(str)).sum())} rows)")
    if bad:
        raise SystemExit(f"{label}: k=all differs from r_fill in {bad}: stop and paste this")
    print(f"check ({label}): k = all equals r_fill on {len(m)} rows, {len(cols)} columns (largest diff {worst:.1e})")


def check_endpoint(tag, folds, seeds):
    files = {"experts": ([], f"experts_{DS_}"), "MI": ([], f"mi_{DS_}_own"), "TMC/ECML": (["method"], f"evidential_{DS_}"),
             "dropout model": ([], f"dropout_{DS_}_sd50")}
    full = [f"io_fill:{s}:k{n}" for s, n in IMUS.items()]
    for label, (by, pre) in files.items():
        new = pd.read_parquet(RES / f"{pre}_{tag}_imuoff.parquet")
        old = pd.read_parquet(RES / f"{pre}_{FULL}_realdrop.parquet")
        old = old[(old.part == "test") & old.condition.str.startswith("r_fill:") & old.fold.isin(folds)
                  & old.seed.isin(seeds)].copy()
        g = old.condition.str.split(":").str[1]
        old["condition"] = "io_fill:" + g + ":k" + g.map(IMUS).astype(str)
        compare(old, new[new.condition.isin(full)], KEYS + by, label)


# ---------------------------------------------------------------- metrics
def fold_metrics(te, D, named, final, t, system, rev):
    """One dict per (group, k) of one fold, plus the clean row (group '', k 0)."""
    g, k = group_k(te)
    y, pred = te.y.to_numpy(), te.pred.to_numpy()
    c0 = k == 0
    rows = []
    for (gg, kk) in [("", 0)] + [(s, j) for s in SENS for j in range(1, IMUS[s] + 1)]:
        m = (g == gg) & (k == kk)
        if not m.any():
            continue
        fl = D[m] >= t
        both = m | c0
        r = dict(system=system, revision=rev, fold=int(te.fold.iloc[0]), unit=unit(te), group=gg or "clean", k=kk,
                 n=int(m.sum()), acc_alone=np.mean(pred[m] == y[m]), acc_policy=np.mean(final[m] == y[m]),
                 flagged=np.mean(fl), auroc=auroc(D[both], m[both]) if kk else np.nan)
        if (named != "").any() and kk:
            r["naming_all"] = np.mean(named[m] == gg)
            r["naming_flagged"] = np.mean(named[m][fl] == gg) if fl.any() else np.nan
        rows.append(r)
    return rows


def answers(raw):
    """Accuracy of the answers that need no certificate, per fold, group and k."""
    g, k = group_k(raw)
    y = raw.y.to_numpy()
    known = np.full(len(raw), -1)
    hidden = np.full(len(raw), -1)
    for s in SENS:
        m = g == s
        known[m] = raw[CP.backup_col(CP.BACKUPS[DS_][s])].to_numpy()[m]
        hidden[m] = raw[f"sd50_hide_pred_{s}"].to_numpy()[m]
    a = pd.DataFrame(dict(fold=raw.fold.to_numpy(), group=np.where(g == "", "clean", g), k=k,
                          nothing=raw.pred.to_numpy() == y, known=np.where(known >= 0, known == y, np.nan),
                          flagged_absent=raw.mask_missing_pred.to_numpy() == y,
                          dropout_alone=raw.sd50_pred.to_numpy() == y,
                          dropout_hidden=np.where(hidden >= 0, hidden == y, np.nan)))
    return a.groupby(["fold", "group", "k"], sort=False).mean().reset_index()


# ---------------------------------------------------------------- predictions (plan A.6)
def predictions(pf):
    m = pf[(pf.system == SYS_MAIN) & (pf.revision == 1) & (pf.group != "clean")].copy()
    m["diff"] = 100 * (m.acc_policy - m.acc_alone)                  # points; per fold (seeds pooled)
    w = m.pivot_table(index=["group", "k"], columns="fold", values="diff")
    legs = w.loc["legs"]
    out, verdict = [], lambda ok: "CONFIRMED" if ok else "NOT CONFIRMED"
    print("\nMAIN RESULT (named in advance): legs, REV3 (main) use / drop rev. 1 minus doing nothing, points,"
          " mean over folds [95% interval], folds better / worse")
    for kk, v in legs.iterrows():
        mm, lo, hi = mci(v.to_numpy())
        print(f"  k = {kk}: {mm:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]  better {int((v > 0).sum())}, worse {int((v < 0).sum())}"
              f" of {v.notna().sum()}")
        out.append(dict(item="main", group="legs", k=kk, mean=mm, lo=lo, hi=hi, n_better=int((v > 0).sum()),
                        n_worse=int((v < 0).sum()), n_folds=int(v.notna().sum())))
    mean_k = legs.mean(axis=1)
    p1 = all(-2.0 <= mean_k.loc[kk] <= 0.0 for kk in (1, 2))
    _, lo7, _ = mci(legs.loc[7].to_numpy())
    p2 = mean_k.loc[7] > 0 and lo7 > 0
    ks = np.array(legs.index, float)
    slopes = np.array([np.polyfit(ks, legs[f].to_numpy(), 1)[0] for f in legs.columns])
    sm, slo, shi = mci(slopes)
    p3 = (slopes > 0).sum() >= 9 and slo > 0
    kstar = next((kk for kk in legs.index if (mean_k.loc[kk:] >= 0).all()), None)
    p4 = kstar is not None and 3 <= kstar <= 6
    other = {(s, kk): w.loc[(s, kk)].mean() for s in ("arms", "trunk") for kk in range(1, IMUS[s] + 1)}
    p5 = all(abs(v) <= 1.5 for v in other.values())
    print("\nPREDICTIONS (plan A.6), each with a plain verdict:")
    print(f"  P1 legs k=1, k=2 mean difference in [-2, 0]: k1 {mean_k.loc[1]:+.2f}, k2 {mean_k.loc[2]:+.2f}"
          f" -> {verdict(p1)}")
    print(f"  P2 legs k=7 mean > 0 with the 95% interval above 0: {mean_k.loc[7]:+.2f}, lower bound {lo7:+.2f}"
          f" -> {verdict(p2)}")
    print(f"  P3 slope over k=1..8 > 0 in >= 9 of 10 folds and interval above 0: {(slopes > 0).sum()} of {len(slopes)}"
          f" folds, mean slope {sm:+.2f} [{slo:+.2f}, {shi:+.2f}] points per IMU -> {verdict(p3)}")
    print(f"  P4 crossover k* in 3..6: k* = {kstar} -> {verdict(p4)}")
    print("  P5 arms and trunk within +/- 1.5 points at every k: "
          + ", ".join(f"{s} k{kk} {v:+.2f}" for (s, kk), v in other.items()) + f" -> {verdict(p5)}")
    for name, ok, val in (("P1", p1, f"k1 {mean_k.loc[1]:+.2f}; k2 {mean_k.loc[2]:+.2f}"),
                          ("P2", p2, f"{mean_k.loc[7]:+.2f}, lower {lo7:+.2f}"),
                          ("P3", p3, f"{(slopes > 0).sum()}/{len(slopes)} folds; slope {sm:+.2f} [{slo:+.2f}, {shi:+.2f}]"),
                          ("P4", p4, f"k* = {kstar}"),
                          ("P5", p5, "; ".join(f"{s} k{kk} {v:+.2f}" for (s, kk), v in other.items()))):
        out.append(dict(item=name, verdict=verdict(ok), value=val))
    for s in ("arms", "trunk"):
        for kk, v in w.loc[s].iterrows():
            mm, lo, hi = mci(v.to_numpy())
            out.append(dict(item="supporting", group=s, k=kk, mean=mm, lo=lo, hi=hi, n_better=int((v > 0).sum()),
                            n_worse=int((v < 0).sum()), n_folds=int(v.notna().sum())))
    return pd.DataFrame(out)


def report(pf, ans):
    pd.set_option("display.width", 230)
    pct = lambda d: (100 * d).round(1)
    print("\nANSWERS WITHOUT A CERTIFICATE, accuracy %, mean over folds")
    t = ans.groupby(["group", "k"], sort=False)[["nothing", "known", "flagged_absent", "dropout_alone",
                                                 "dropout_hidden"]].mean()
    print(pct(t).to_string())
    for rev in (1, 2):
        q = pf[pf.revision == rev]
        print(f"\nSYSTEMS, revision {rev}" + (" (post hoc, labelled)" if rev == 2 else "")
              + ": use / drop accuracy %, mean over folds (clean row = clean clips)")
        print(pct(q.pivot_table(index=["group", "k"], columns="system", values="acc_policy", sort=False)).to_string())
        if rev == 1:
            print("\n  flagged % (clean row = false alarms)")
            print(pct(q.pivot_table(index=["group", "k"], columns="system", values="flagged", sort=False)).to_string())
            print("\n  detection AUROC, IMU-off clips vs the fold's clean clips")
            print(q[q.k > 0].pivot_table(index=["group", "k"], columns="system", values="auroc", sort=False)
                  .round(3).to_string())
            print("\n  names the right group, % of flagged clips")
            nm = q[(q.k > 0) & (q.system != SYS_CONF)]
            print(pct(nm.pivot_table(index=["group", "k"], columns="system", values="naming_flagged", sort=False))
                  .to_string())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=FULL, help=f"{FULL} = full run; e.g. f0_s0 = smoke-run files (code check)")
    args = ap.parse_args(); ds = DS_
    if ds not in CP.BACKUPS:
        raise SystemExit("configs/backups.json has no d3 entry")
    MS.journal_files(ds)

    b = bench(args.tag)
    folds, seeds = sorted(b.fold.unique()), sorted(b.seed.unique())
    keep = lambda d: d[d.fold.isin(folds) & d.seed.isin(seeds)]
    lie = keep(MS.add_mi(CP.load(ds), ds))
    lie = lie[lie.available]
    check_clean(lie, b, ["pred", "msp", "exp_js", "s4"] + [f"exp_support_{m}" for m in CP.EXP[ds]], f"{ds} B4")
    ev_l = keep(pd.read_parquet(RES / f"{EVFILE[ds]}.parquet")); ev_l = ev_l[ev_l.available]
    ev_b = pd.read_parquet(RES / f"evidential_{ds}_{args.tag}_imuoff.parquet")
    for m in MS.MAIN_EV[ds]:
        check_clean(ev_l[ev_l.method == m], ev_b[ev_b.method == m], ["pred", "msp", "u", "conflict"], f"{ds} {m}")
    d_l = with_dropout(lie, RES / f"dropout_{ds}_sd50_{FULL}.parquet", ds)
    raw = b.copy()
    d_b = with_dropout(b, RES / f"dropout_{ds}_sd50_{args.tag}_imuoff.parquet", ds)
    check_clean(d_l, d_b, ["pred", "msp"], f"{ds} dropout sd50")
    raw = raw.merge(d_b[KEYS + ["sd50_pred"] + [f"sd50_hide_pred_{g}" for g in SENS]], on=KEYS, validate="one_to_one")
    check_endpoint(args.tag, folds, seeds)

    rev3 = MS.signals(REV3_SIG)
    systems = {
        SYS_CONF: (lie, b, MS.signals(["s1"]), None, None),
        SYS_C3: (lie, b, MS.signals(["s1", "s2", "s3"]), {"odd one out": MS.ODD}, CP.backup_answer),
        SYS_MAIN: (lie, b, rev3, REV3_RULES, CP.backup_answer),
        **{f"{m.upper()} alone": (ev_l[ev_l.method == m], ev_b[ev_b.method == m], ev_feats, EV_RULES, ev_drop)
           for m in MS.MAIN_EV[ds]},
        SYS_D: (d_l, d_b, rev3, REV3_RULES, DSC.hide_answer("sd50_hide_pred")),
    }
    rows, rev3_rows = [], []
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
                rows += fold_metrics(te, D, named, final, fz.t, name_, rev)
                if name_ == SYS_MAIN and rev == 1:
                    g, k = group_k(te)
                    rev3_rows.append(pd.DataFrame(dict(fold=te.fold.to_numpy(), seed=te.seed.to_numpy(),
                                                       clip=te["clip"].to_numpy(), group=g, k=k, y=te.y.to_numpy(),
                                                       pred=te.pred.to_numpy(), final=final, score=D,
                                                       flagged=D >= fz.t, named=named)))
        print(f"scored: {name_}" + (" (frozen fit reproduces the lie numbers exactly)" if rules else ""))
    pf = pd.DataFrame(rows)
    n_rows = pf.groupby(["system", "revision"]).n.sum()
    if n_rows.nunique() != 1:
        raise SystemExit(f"systems scored on different numbers of rows: {n_rows.to_dict()}: stop and paste this")
    print(f"check: every system scored on the same {int(n_rows.iloc[0])} rows "
          f"({len(folds)} folds, {len(seeds)} seeds, {pf.k.nunique() - 1} IMU-off conditions + clean)")

    ans = answers(raw)
    report(pf, ans)
    pr = predictions(pf) if len(folds) > 2 else None
    if pr is None:
        print("\nsmoke run: fewer than 3 folds, predictions not evaluated (code check only)")
    sfx = "" if args.tag == FULL else f"_{args.tag}"
    pf.to_csv(RES / f"imuoff_{ds}{sfx}_per_fold.csv", index=False)
    ans.to_csv(RES / f"imuoff_{ds}{sfx}_answers_per_fold.csv", index=False)
    pd.concat(rev3_rows, ignore_index=True).to_parquet(RES / f"imuoff_{ds}{sfx}_rev3_rows.parquet", index=False)
    if pr is not None:
        pr.to_csv(RES / f"imuoff_{ds}{sfx}_predictions.csv", index=False)
    print(f"\nsaved: results/certificate/imuoff_{ds}{sfx}_per_fold.csv, _answers_per_fold.csv, _rev3_rows.parquet"
          + (", _predictions.csv" if pr is not None else ""))


if __name__ == "__main__":
    main()
