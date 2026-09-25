"""Phase E, step 2: score TMC and ECML exactly like our certificate, side by side with it.

Reads results/certificate/evidential_<ds>...parquet (cert_evidential.py; both settings when present)
and our certificate's own inputs through cert_policy.load (the files behind v3/v4/v5).
Everything is fitted on each fold's VALIDATION person and applied unchanged to its TEST person;
one number per test subject, then means and paired differences (ours minus theirs) across subjects.

Version choice, fixed before any result (2026-09-25): for each method the main comparison uses the
setting (our protocol, or the paper's optimiser settings "_p") whose COMBINED score separates lies from
benign clips better on the validation people (mean validation AUROC). The other is kept in the CSV.

  LIE     COMBINED score fitted on validation (ours: 1-msp, expert JS, lowest expert support;
          theirs: 1-msp, u, conflict); positives s_swap:*, negatives clean / c_swap_same:* / h_all:*
  ERROR   clean rows, each system's OWN mistakes (>= 5 mistakes and >= 5 correct per person)
  NAMING  UTD-MHAD lie rows (chance 1/3). Ours: odd one out, as in cert_policy.py (the validation choice
          in all 8 folds). Theirs: best of three rules on validation (most uncertain / most in conflict /
          least support). MMAct: not scored (two sensors)
  POLICY  use / drop / abstain, revisions 1 and 2 exactly as cert_policy.py. Their "drop" leaves the
          named sensor out of their own fusion. Our numbers are checked against cert_policy.evaluate.
Writes results/certificate/e_<ds>_per_subject.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cert_policy as CP
from cert_score import auroc, fit_logreg, tq, MIN_CLASS

RES = ROOT / "results/certificate"
SENS = CP.EXP                                     # same sensor order as cert_policy.py (matters only for ties)
EVFILE = {"d1": "evidential_d1_all_s0-1-2_att-imu-skel-video", "d2": "evidential_d2_all_s0-1-2"}
EV_RULES = {"most uncertain": ("ev_u", "max"), "most in conflict": ("ev_cdeg", "max"),
            "least supports the answer": ("ev_supp", "min")}
OUR_RULES = {"odd one out": ("exp_div", "max")}
COV = (1.0, 0.9, 0.8)


def family(name):
    return name.split("_")[0].upper()


def is_lie(d):
    return d.condition.str.startswith("s_swap").to_numpy()


def ev_feats(d, ds):
    return np.column_stack([1 - d.msp.to_numpy(), d.u.to_numpy(), d.conflict.to_numpy()])


def ev_drop(d, ds, named):
    out = np.empty(len(d), dtype=int)
    for s in SENS[ds]:
        k = named == s
        out[k] = d[f"drop_pred_{s}"].to_numpy()[k]
    return out


def fit_apply(Xv, yv, *others):
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
    w = fit_logreg((Xv - mu) / sd, np.asarray(yv, float))
    return [np.c_[np.ones(len(X)), (X - mu) / sd] @ w for X in (Xv, *others)]


def name(d, ds, rules, rule):
    col, how = rules[rule]
    v = d[[f"{col}_{m}" for m in SENS[ds]]].to_numpy()
    return np.array(SENS[ds])[v.argmax(1) if how == "max" else v.argmin(1)]


def per_subject(df, ds, feats, rules, drop_answer, rev):
    rows = []
    for fold, g in df[df.available].groupby("fold"):
        va, te = g[g.part == "val"], g[g.part == "test"]
        lv, lt = is_lie(va), is_lie(te)
        Xv, Xt = feats(va, ds), feats(te, ds)
        cv, ct = (va.condition == "clean").to_numpy(), (te.condition == "clean").to_numpy()
        yv, yt, fv, ft = va.y.to_numpy(), te.y.to_numpy(), va.pred.to_numpy(), te.pred.to_numpy()
        r = dict(fold=fold, subject=te.subject.iloc[0])
        Dv, Dt = fit_apply(Xv, lv, Xt)                                   # LIE: combined score
        r.update(val_lie_auroc=auroc(Dv, lv), lie_auroc=auroc(Dt, lt), lie_auroc_msp=auroc(Xt[:, 0], lt))
        ev_, et = fv[cv] != yv[cv], ft[ct] != yt[ct]                     # ERROR: own mistakes, clean rows
        ok = ev_.sum() >= MIN_CLASS and (~ev_).sum() >= MIN_CLASS
        r["error_auroc"] = auroc(fit_apply(Xv[cv], ev_, Xt[ct])[1], et, MIN_CLASS) if ok else np.nan
        r["error_auroc_msp"] = auroc(Xt[ct, 0], et, MIN_CLASS)
        if ds == "d1":                                                   # NAMING, then use / drop
            L = va[lv]
            acc = {q: np.mean(name(L, ds, rules, q) == L.attacked.to_numpy()) for q in rules}
            rule = max(acc, key=acc.get)
            r["rule"] = rule
            r["naming"] = np.mean(name(te[lt], ds, rules, rule) == te[lt].attacked.to_numpy())
            bv = drop_answer(va, ds, name(va, ds, rules, rule))
            bt = drop_answer(te, ds, name(te, ds, rules, rule))
            if rev == 1:                                                 # as cert_policy.py revision 1
                cand = np.r_[np.inf, np.quantile(Dv, np.linspace(1, 0, 101))]
                t = cand[int(np.argmax([np.mean(np.where(Dv >= c, bv, fv) == yv) for c in cand]))]
            else:                                                        # as cert_policy.py revision 2
                t = np.nextafter(np.quantile(Dv[cv], 1 - CP.BUDGET), np.inf)
            final_v, final_t = np.where(Dv >= t, bv, fv), np.where(Dt >= t, bt, ft)
        else:                                                            # MMAct: use / abstain only
            r["rule"], r["naming"] = "", np.nan
            final_v, final_t = fv, ft
        At = Dt if rev == 1 else fit_apply(Xv, final_v != yv, Xt)[1]    # abstain score
        grp = CP.group(te.condition)
        for gname in ("normal", "lie", "benign change"):
            k = grp == gname
            r[f"acc_main_{gname}"], r[f"acc_policy_{gname}"] = np.mean(ft[k] == yt[k]), np.mean(final_t[k] == yt[k])
        for cov in COV:
            r[f"all_cov{int(cov * 100)}_cert_policy"] = CP.selective(At, final_t, yt, cov)
            r[f"all_cov{int(cov * 100)}_msp_main"] = CP.selective(1 - te.msp.to_numpy(), ft, yt, cov)
        rows.append(r)
    return pd.DataFrame(rows).set_index("subject")


def ci(v, k=1.0, d=1):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v)
    if n < 2:
        return "n/a"
    m, h = v.mean(), tq(n - 1) * v.std(ddof=1) / np.sqrt(n)
    return f"{k * m:+.{d}f} [{k * (m - h):+.{d}f}, {k * (m + h):+.{d}f}] {int((v > 0).sum())}/{n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("d1", "d2"), required=True)
    ds = ap.parse_args().dataset
    files = [RES / f"{EVFILE[ds]}{s}.parquet" for s in ("", "_paper")]
    ev = pd.concat([pd.read_parquet(f) for f in files if f.exists()], ignore_index=True)
    ours = CP.load(ds)
    per = {}
    for rev in (1, 2):
        per[("ours", rev)] = p = per_subject(ours, ds, CP.features, OUR_RULES, CP.backup_answer, rev)
        CP.REV = rev
        ref = CP.evaluate(ours, ds, "odd" if ds == "d1" else None, "check").set_index("subject")
        dev = max(float(np.nanmax(np.abs(p[c] - ref[c])))
                  for c in ("acc_policy_lie", "acc_policy_normal", "all_cov80_cert_policy"))
        if dev > 1e-9:
            raise SystemExit(f"our recomputed policy differs from cert_policy.py by {dev:.2e}: stop and paste this")
        for m, d in ev.groupby("method"):
            per[(m, rev)] = per_subject(d, ds, ev_feats, EV_RULES, ev_drop, rev)
    print("check: our certificate's numbers reproduce cert_policy.py exactly (revisions 1 and 2)")

    names = sorted(ev.method.unique())
    chosen = {}
    print("VERSION CHOICE (mean validation lie AUROC; rule fixed before any result):")
    for f in sorted({family(n) for n in names}):
        c = {n: per[(n, 1)].val_lie_auroc.mean() for n in names if family(n) == f}
        chosen[f] = max(c, key=c.get)
        print(f"  {f}: " + ", ".join(f"{n} {v:.3f}" for n, v in c.items()) + f"  -> {chosen[f]}")
    systems = [("ours (certificate)", "ours")] + [(f"{f} ({n})", n) for f, n in chosen.items()]

    o = per[("ours", 1)]
    print(f"\n{ds}: DETECT and NAME (mean over test people)")
    print(f"  {'system':20s} lie_AUROC  error_AUROC  naming%")
    for label, key in systems:
        q = per[(key, 1)]
        nm = f"{100 * q.naming.mean():7.1f}" if ds == "d1" else "    n/a"
        print(f"  {label:20s} {q.lie_auroc.mean():9.3f} {q.error_auroc.mean():12.3f} {nm}")
    for label, key in systems[1:]:
        q = per[(key, 1)]
        print(f"  ours minus {label}: lie {ci(o.lie_auroc - q.lie_auroc, 1, 3)} | error {ci(o.error_auroc - q.error_auroc, 1, 3)}"
              + (f" | naming {ci(o.naming - q.naming, 100)}" if ds == "d1" else ""))
    for rev in (1, 2):
        o = per[("ours", rev)]
        print(f"\n{ds}: USE / DROP / ABSTAIN, revision {rev}" + (" (post hoc, labelled)" if rev == 2 else "")
              + ("" if ds == "d1" else "  [MMAct: use / abstain only]"))
        print(f"  {'system':20s} lie_acc%  normal_acc%  answer80%")
        for label, key in systems:
            q = per[(key, rev)]
            print(f"  {label:20s} {100 * q.acc_policy_lie.mean():8.1f} {100 * q.acc_policy_normal.mean():12.1f} "
                  f"{100 * q.all_cov80_cert_policy.mean():10.1f}")
        for label, key in systems[1:]:
            q = per[(key, rev)]
            print(f"  ours minus {label}: lie {ci(o.acc_policy_lie - q.acc_policy_lie, 100)} | normal "
                  f"{ci(o.acc_policy_normal - q.acc_policy_normal, 100)} | answer80 "
                  f"{ci(o.all_cov80_cert_policy - q.all_cov80_cert_policy, 100)}")
    out = pd.concat([v.assign(system=k[0], revision=k[1]) for k, v in per.items()]).reset_index()
    out.to_csv(RES / f"e_{ds}_per_subject.csv", index=False)
    print(f"\nsaved: results/certificate/e_{ds}_per_subject.csv")


if __name__ == "__main__":
    main()
