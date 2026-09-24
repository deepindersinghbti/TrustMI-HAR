"""Certificate v1, step 2: how well do the label-free signs work?

Every choice is fitted on each fold's VALIDATION person and applied unchanged to that fold's TEST person:
the harmless strength level, the weights of the combined score, and the sensor-naming rule.
The test person (one per fold) is the unit: one AUROC per test subject, then mean [95% CI] across subjects.

  Task LIE    positives: s_swap:* rows; negatives: clean, c_swap_same:*, h_all:* rows (all seeds pooled)
  Task ERROR  clean rows only; positives: the model's natural mistakes
              (a subject counts only with >= 5 mistakes and >= 5 correct, pooled over seeds)
  NAMING      MMAct s_swap rows: is the swapped sensor named? (chance 50%). Not scored on UTD-MHAD,
              where only the IMU is ever swapped, so naming would be trivially "the IMU".
  --with-experts  also merges experts_<ds>_all_s0-1-2.parquet (cert_experts.py) and adds the expert signs,
                  a second combined score and two expert naming rules; writes v2_*.csv instead of v1_*.csv.
                  Without the flag the output is exactly the v1 analysis.
Writes results/certificate/v1_*.csv (or v2_*.csv)
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/certificate"
FILES = {"d1": RES / "signals_d1_all_s0-1-2.parquet", "d2": RES / "signals_d2_all_s0-1-2.parquet"}
MODS = {"d1": ["video", "skel", "imu"], "d2": ["watch", "phone"]}
SENS = {"d1": ["imu"], "d2": ["watch", "phone"]}
LEVELS, DEFAULT_LEVEL, MIN_CLASS = (1, 2, 3), 2, 5
BASE = "1 - msp (baseline)"
COMB_INPUTS = [BASE, "S1 stability drop", "S2 hide-one shift"]
EXPERT_MODS = {"d1": ["imu", "skel", "video"], "d2": ["watch", "phone"]}
COMB2_INPUTS = [BASE, "X experts disagree (JS)", "X lowest expert support"]
KEYS = ["fold", "part", "seed", "clip", "condition"]


def tq(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        return {7: 2.365, 19: 2.093}.get(df, 1.96)


def fmt(v, k=1.0, d=3):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v)
    if n == 0:
        return "n/a"
    m = v.mean(); h = tq(n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return f"{k * m:.{d}f} [{k * (m - h):.{d}f}, {k * (m + h):.{d}f}] (n={n})"


def auroc(score, label, min_class=1):
    label = np.asarray(label, bool); n1, n0 = label.sum(), (~label).sum()
    if n1 < min_class or n0 < min_class:
        return np.nan
    r = pd.Series(np.asarray(score, float)).rank().to_numpy()      # average ranks: ties count half
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(d, ds, L):
    """Label-free suspicion scores; higher = more suspicious."""
    s, m = SENS[ds], MODS[ds]
    out = {
        BASE: 1 - d.msp.to_numpy(),
        "entropy (baseline)": d.entropy.to_numpy(),
        "energy (baseline)": d.energy.to_numpy(),
        "S1 stability flip": d[[f"stab_flip_{x}_L{L}" for x in s]].mean(axis=1).to_numpy(),
        "S1 stability drop": d[[f"stab_drop_{x}_L{L}" for x in s]].mean(axis=1).to_numpy(),
        "S2 hide-one shift": d[[f"hide_tv_{x}" for x in m]].max(axis=1).to_numpy(),
        "S2 hide-one flips": d[[f"hide_flip_{x}" for x in m]].mean(axis=1).to_numpy(),
    }
    if "exp_js" in d.columns:                        # expert signs (only with --with-experts)
        sup = d[[f"exp_support_{x}" for x in EXPERT_MODS[ds]]].to_numpy()
        out["X experts disagree (JS)"] = d.exp_js.to_numpy()
        out["X lowest expert support"] = 1 - sup.min(1)
        out["X mean expert support"] = 1 - sup.mean(1)
    return out


def fit_logreg(X, y, lam=1.0, iters=50):
    """L2-regularised logistic regression by Newton steps (intercept not penalised)."""
    X1 = np.c_[np.ones(len(X)), X]; w = np.zeros(X1.shape[1]); P = np.diag(np.r_[0.0, np.ones(X.shape[1])])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (p - y) + lam * P @ w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + lam * P
        w -= np.linalg.solve(H, g)
    return w


def task_rows(d, task):
    d = d[d.available]
    if task == "LIE":
        return d, d.condition.str.startswith("s_swap").to_numpy()
    d = d[d.condition == "clean"]
    return d, ~d.correct.to_numpy()


def score_task(df, ds, task):
    rows, levels = [], []
    for fold, g in df.groupby("fold"):
        va, yv = task_rows(g[g.part == "val"], task)
        te, yt = task_rows(g[g.part == "test"], task)
        val_auc = {L: auroc(scores(va, ds, L)["S1 stability drop"], yv, MIN_CLASS) for L in LEVELS}
        ok = {L: a for L, a in val_auc.items() if not np.isnan(a)}
        L = max(ok, key=ok.get) if ok else DEFAULT_LEVEL
        levels.append(L)
        sv, st = scores(va, ds, L), scores(te, ds, L)
        combos = [("COMBINED (S1+S2+msp)", COMB_INPUTS)]
        if "X experts disagree (JS)" in sv:
            combos.append(("COMBINED v2 (experts+msp)", COMB2_INPUTS))
        for cname, inputs in combos:
            if yv.sum() >= MIN_CLASS and (~yv).sum() >= MIN_CLASS:
                Xv = np.column_stack([sv[k] for k in inputs]); mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
                w = fit_logreg((Xv - mu) / sd, yv.astype(float))
                st[cname] = np.c_[np.ones(len(te)), (np.column_stack([st[k] for k in inputs]) - mu) / sd] @ w
            else:
                st[cname] = np.full(len(te), np.nan)
        min_c = MIN_CLASS if task == "ERROR" else 1
        for name, sc in st.items():
            rows.append(dict(dataset=ds, task=task, fold=fold, subject=te.subject.iloc[0], score=name,
                             auroc=auroc(sc, yt, min_c) if not np.isnan(sc).all() else np.nan, level=L))
    return pd.DataFrame(rows), levels


def summarise(res):
    out = []
    for (ds, task), g in res.groupby(["dataset", "task"], sort=False):
        w = g.pivot_table(index="subject", columns="score", values="auroc", dropna=False)
        for name in w.columns:
            diff = (w[name] - w[BASE]).dropna()
            out.append(dict(dataset=ds, task=task, score=name, auroc=fmt(w[name]),
                            minus_msp=fmt(diff) if name != BASE else "",
                            subjects_better=f"{int((diff > 0).sum())}/{len(diff)}" if name != BASE else ""))
    return pd.DataFrame(out)


def naming(df, ds):
    """MMAct only: choose the rule on each fold's validation s_swap rows, apply to its test rows."""
    mods = np.array(MODS[ds])
    rules = {
        "most support (argmax hide_drop)": lambda d, L: d[[f"hide_drop_{m}" for m in mods]].to_numpy().argmax(1),
        "least support (argmin hide_drop)": lambda d, L: d[[f"hide_drop_{m}" for m in mods]].to_numpy().argmin(1),
        "most confident without it (argmax hide_msp)": lambda d, L: d[[f"hide_msp_{m}" for m in mods]].to_numpy().argmax(1),
        "least stable sensor (argmax stab_drop)": lambda d, L: d[[f"stab_drop_{m}_L{L}" for m in mods]].to_numpy().argmax(1),
    }
    if "exp_js" in df.columns:
        rules["expert least supporting the answer (argmin exp_support)"] = \
            lambda d, L: d[[f"exp_support_{m}" for m in mods]].to_numpy().argmin(1)
        rules["least confident expert (argmin exp_conf)"] = \
            lambda d, L: d[[f"exp_conf_{m}" for m in mods]].to_numpy().argmin(1)
    rows = []
    for fold, g in df[df.available & df.condition.str.startswith("s_swap")].groupby("fold"):
        va, te = g[g.part == "val"], g[g.part == "test"]
        acc_val = {r: (mods[f(va, DEFAULT_LEVEL)] == va.attacked.to_numpy()).mean() for r, f in rules.items()}
        best = max(acc_val, key=acc_val.get)
        for r, f in rules.items():
            rows.append(dict(fold=fold, subject=te.subject.iloc[0], rule=r, chosen=(r == best),
                             test_acc=(mods[f(te, DEFAULT_LEVEL)] == te.attacked.to_numpy()).mean()))
    return pd.DataFrame(rows)


def load(ds, with_experts):
    df = pd.read_parquet(FILES[ds])
    if not with_experts:
        return df
    ex = pd.read_parquet(RES / f"experts_{ds}_all_s0-1-2.parquet")
    m = df.merge(ex, on=KEYS, how="inner", validate="one_to_one")
    if len(m) != len(df):
        raise SystemExit(f"{ds}: {len(df) - len(m)} signal rows have no matching expert row")
    agree = float((m.pred == m.pred_fused).mean())
    print(f"{ds}: experts merged; fused answers identical in {agree:.4%} of {len(m)} rows")
    if agree < 0.999:
        raise SystemExit("the two runs did not see the same inputs: stop and paste this")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-experts", action="store_true")
    args = ap.parse_args()
    pre = "v2" if args.with_experts else "v1"
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    allres, lv, data = [], {}, {}
    for ds in FILES:
        df = data[ds] = load(ds, args.with_experts)
        for task in ("LIE", "ERROR"):
            r, levels = score_task(df, ds, task)
            allres.append(r); lv[(ds, task)] = pd.Series(levels).value_counts().sort_index().to_dict()
    res = pd.concat(allres, ignore_index=True)
    summ = summarise(res)
    print("AUROC per test subject, mean [95% CI]; minus_msp = paired difference to the 1 - msp baseline")
    print(summ.to_string(index=False))
    print("\nharmless level chosen on validation (level: folds):", {f"{k[0]} {k[1]}": v for k, v in lv.items()})
    nm = naming(data["d2"], "d2")
    chosen = nm[nm.chosen].groupby("subject").test_acc.mean()
    print(f"\nNAMING the swapped sensor, MMAct (chance 50%): rule chosen on validation -> "
          f"test accuracy {fmt(chosen, 100, 1)}")
    print("rules chosen per fold:", nm[nm.chosen].rule.value_counts().to_dict())
    print("each rule on test, for transparency only (not used for any choice):")
    print(nm.groupby("rule").test_acc.apply(lambda v: fmt(v, 100, 1)).to_string())
    res.to_csv(RES / f"{pre}_auroc_per_subject.csv", index=False)
    summ.to_csv(RES / f"{pre}_auroc_summary.csv", index=False)
    nm.to_csv(RES / f"{pre}_naming_per_subject.csv", index=False)
    print(f"\nsaved: {RES.relative_to(ROOT)}/{pre}_*.csv")


if __name__ == "__main__":
    main()
