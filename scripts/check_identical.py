"""Are two result parquets the same, row by row? (plan step 2: refactor check; reusable later)

  python scripts/check_identical.py OLD NEW [--fold F] [--by method]
OLD may cover more folds than NEW (--fold restricts OLD). Rows are matched on fold, part, seed, clip,
condition, plus any --by columns (needed for files with several rows per clip, e.g. evidential_*: one row
per method). Columns present in both are compared: exactly for ids, labels, answers and flags; floats are
reported as the largest absolute difference and pass below --tol. Prints IDENTICAL or the differences.
"""
import argparse, sys
import numpy as np
import pandas as pd

KEYS = ["fold", "part", "seed", "clip", "condition"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--by", default="", help="extra comma-separated key columns, e.g. method")
    a = ap.parse_args()
    keys = KEYS + [c for c in a.by.split(",") if c]
    old, new = pd.read_parquet(a.old), pd.read_parquet(a.new)
    if a.fold is not None:
        old = old[old.fold == a.fold]
    old = old[old.seed.isin(new.seed.unique())]
    print(f"rows: old {len(old)}, new {len(new)}")
    only_old, only_new = sorted(set(old.columns) - set(new.columns)), sorted(set(new.columns) - set(old.columns))
    if only_old or only_new:
        print(f"columns only in old (not compared): {only_old}\ncolumns only in new (not compared): {only_new}")
    for name, d in (("old", old), ("new", new)):        # one row per key, or the comparison is meaningless
        dup = int(d.duplicated(keys).sum())
        if dup:
            print(f"{name} file has {dup} rows with a repeated key {keys}: add --by (e.g. --by method)")
            sys.exit(1)
    m = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    unmatched = (m["_merge"] != "both").sum()
    bad = []
    if unmatched:
        bad.append(f"{unmatched} rows exist in only one file")
    m = m[m["_merge"] == "both"]
    worst = 0.0
    for c in sorted(set(old.columns) & set(new.columns) - set(keys)):
        x, y = m[f"{c}_old"].to_numpy(), m[f"{c}_new"].to_numpy()
        if np.issubdtype(x.dtype, np.floating) and np.issubdtype(y.dtype, np.floating):
            d = float(np.nanmax(np.abs(x - y))) if len(x) else 0.0
            same_nan = np.array_equal(np.isnan(x), np.isnan(y))
            worst = max(worst, d)
            if d > a.tol or not same_nan:
                bad.append(f"{c}: max |diff| {d:.2e}" + ("" if same_nan else " (NaN pattern differs)"))
        elif not np.array_equal(x.astype(str), y.astype(str)):
            bad.append(f"{c}: {(x.astype(str) != y.astype(str)).sum()} rows differ")
    print(f"compared {len(m)} matched rows; largest float difference {worst:.2e}")
    if bad:
        print("DIFFERENT:\n  " + "\n  ".join(bad)); sys.exit(1)
    print("IDENTICAL")


if __name__ == "__main__":
    main()
