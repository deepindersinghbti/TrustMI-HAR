"""A3 statistics with the SUBJECT as the unit of analysis (blueprint). Read-only on existing outputs.

  Table 1  every operator: mean [95% CI] across subjects of flip rate, accuracy, confidence
  Table 2  the key paired contrast: different-action swap minus same-action swap, per subject
  Check 1  do same-action / harmless flips happen on clips the model already got wrong?
  Check 2  does A1's raw swap ratio (same/different, per subject) predict how reliable that sensor is?
Writes results/operators/A3_subject_level_*.csv
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/operators"
FILES = {"d1": RES / "A3_d1_b4_all_s0-1-2.parquet", "d2": RES / "A3_d2_b4s_all_s0-1-2.parquet"}
A1_LOG = ROOT / "logs/operators_check_allfolds.log"
UNIMODAL = {("d1", "imu"): "b1_imu", ("d2", "watch"): "b1_watch", ("d2", "phone"): "b1_phone"}
SEEDS = (0, 1, 2)


def tq(df):
    """97.5% quantile of Student's t (scipy if present, else exact table values for our two cases)."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        return {7: 2.365, 19: 2.093}.get(df, 1.96)


def mean_ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v)
    m = v.mean()
    h = tq(n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - h, m + h, n


def fmt(v, pct=True):
    m, lo, hi, _ = mean_ci(v)
    if pct:
        return f"{100 * m:5.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}]"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def per_subject(df):
    keys = ["dataset", "target", "op", "level", "subject"]
    return df.groupby(keys, sort=False).agg(
        flip=("flipped", "mean"), acc=("correct", "mean"),
        follows=("follows_donor", "mean"), conf=("conf", "mean")).reset_index()


def table1(ps):
    rows = []
    for (ds, t, op, lv), g in ps.groupby(["dataset", "target", "op", "level"], sort=False):
        rows.append(dict(dataset=ds, target=t, op=op, level=lv or "", subjects=len(g),
                         flip=fmt(g.flip), acc=fmt(g.acc),
                         follows_donor=fmt(g.follows) if op == "s_swap" else "",
                         conf=fmt(g.conf, pct=False)))
    return pd.DataFrame(rows)


def table2(ps):
    rows = []
    for (ds, t), g in ps.groupby(["dataset", "target"], sort=False):
        w = g.pivot_table(index="subject", columns="op", values="flip")
        d = w["s_swap"] - w["c_swap_same"]
        rows.append(dict(dataset=ds, target=t, subjects=len(d),
                         s_swap=fmt(w["s_swap"]), c_swap_same=fmt(w["c_swap_same"]),
                         difference=fmt(d), subjects_positive=f"{int((d > 0).sum())}/{len(d)}",
                         smallest_difference=f"{100 * d.min():.1f}"))
    return pd.DataFrame(rows)


def check1(df):
    """Flip rate split by whether the CLEAN prediction was right; per subject, then mean [CI]."""
    rows = []
    sel = df[(df.op == "c_swap_same") | ((df.op == "h_all") & (df.level == 3))]
    for (ds, t, op), g in sel.groupby(["dataset", "target", "op"], sort=False):
        s = g.groupby(["subject", "correct_clean"]).flipped.mean().unstack()
        rows.append(dict(dataset=ds, target=t, op=op + (" L3" if op == "h_all" else ""),
                         flip_if_clean_correct=fmt(s.get(True, pd.Series(dtype=float))),
                         flip_if_clean_wrong=fmt(s.get(False, pd.Series(dtype=float))),
                         acc_change=fmt(g.groupby("subject").correct.mean()
                                        - g.groupby("subject").correct_clean.mean())))
    return pd.DataFrame(rows)


def a1_ratios(log):
    cur, out = None, {}
    for line in open(log):
        m = re.match(r"=== D(\d) (\w+) f(\d+):", line)
        if m:
            cur = (f"d{m.group(1)}", m.group(2), int(m.group(3))); continue
        m = re.match(r"\s+(s_swap|c_swap_same)\s+\S+\s+([0-9.]+)", line)
        if m and cur:
            out.setdefault(cur, {})[m.group(1)] = float(m.group(2))
    return pd.DataFrame([dict(dataset=k[0], target=k[1], fold=k[2], ratio=v["c_swap_same"] / v["s_swap"])
                         for k, v in out.items() if len(v) == 2])


def unimodal_acc(ds, target, fold):
    accs = []
    for s in SEEDS:
        f = ROOT / "runs" / f"BASE_{ds}_{UNIMODAL[(ds, target)]}_loso_f{fold:02d}_s{s}" / "preds_test.parquet"
        if f.exists():
            p = pd.read_parquet(f, columns=["true", "pred"])
            accs.append((p["true"] == p["pred"]).mean())
    return np.mean(accs) if accs else np.nan


def spearman_perm(x, y, n=20000, seed=0):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    r = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return r, float((np.abs(null) >= abs(r) - 1e-12).mean())


def check2(df, ratios):
    sep = (df.groupby(["dataset", "target", "fold", "op"]).flipped.mean().unstack())
    sep = (sep["s_swap"] - sep["c_swap_same"]).rename("separation").reset_index()
    m = ratios.merge(sep, on=["dataset", "target", "fold"])
    m["unimodal_acc"] = [unimodal_acc(r.dataset, r.target, r.fold) for r in m.itertuples()]
    rows = []
    for (ds, t), g in m.groupby(["dataset", "target"], sort=False):
        g = g.dropna()
        r1, p1 = spearman_perm(g.ratio, g.unimodal_acc)
        r2, p2 = spearman_perm(g.ratio, g.separation)
        rows.append(dict(dataset=ds, target=t, subjects=len(g),
                         rho_ratio_vs_unimodal_acc=f"{r1:+.2f} (p={p1:.3f})",
                         rho_ratio_vs_model_separation=f"{r2:+.2f} (p={p2:.3f})"))
    return pd.DataFrame(rows), m


def main():
    df = pd.concat([pd.read_parquet(f) for f in FILES.values()], ignore_index=True)
    ps = per_subject(df)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    t1, t2, c1 = table1(ps), table2(ps), check1(df)
    print("TABLE 1: per operator, mean [95% CI] across subjects (flip/acc/follows in %, conf 0-1)")
    print(t1.to_string(index=False))
    print("\nTABLE 2: flips, different-action swap vs same-action swap (paired by subject, %)")
    print(t2.to_string(index=False))
    print("\nCHECK 1: are flips confined to clips the model already got wrong? (%)")
    print(c1.to_string(index=False))
    c2, m = check2(df, a1_ratios(A1_LOG))
    print("\nCHECK 2: Spearman rho across subjects; prediction: both NEGATIVE"
          " (higher raw same/different ratio -> less reliable sensor)")
    print(c2.to_string(index=False))
    for name, t in (("table1", t1), ("table2", t2), ("check1", c1), ("check2", c2), ("check2_per_subject", m)):
        t.to_csv(RES / f"A3_subject_level_{name}.csv", index=False)
    print(f"\nsaved: {RES.relative_to(ROOT)}/A3_subject_level_*.csv")


if __name__ == "__main__":
    main()
