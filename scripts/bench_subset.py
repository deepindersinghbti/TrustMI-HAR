"""Do the rows a NEW bench shares with an OLD one (same fold, part, seed, clip, condition) agree exactly?
(docs/plan_additions_2026-11-14.md, Checks 2: an addition's bench reruns some old conditions; they must
reproduce the stored files.)

  python scripts/bench_subset.py OLD NEW [--by method]     (or import check_subset)
Only conditions present in OLD are taken from NEW, and only the folds and seeds of NEW from OLD. Every NEW row of
those conditions must have an OLD row. Columns in both are compared: exactly for ids, labels and answers; floats
below --tol (default 1e-6, as check_identical.py and check_clean).
"""
import argparse, sys
import numpy as np
import pandas as pd

KEYS = ["fold", "part", "seed", "clip", "condition"]


def check_subset(old, new, by=(), label="", tol=1e-6):
    keys = KEYS + list(by)
    new = new[new.condition.isin(set(old.condition))]
    old = old[old.fold.isin(new.fold.unique()) & old.seed.isin(new.seed.unique())
              & old.condition.isin(set(new.condition))]
    if len(new) == 0:
        raise SystemExit(f"{label}: no shared conditions between the two benches: stop and paste this")
    m = new.merge(old, on=keys, how="left", suffixes=("_n", "_o"), indicator=True, validate="one_to_one")
    if (m["_merge"] != "both").any() or len(old) != len(new):
        raise SystemExit(f"{label}: {int((m['_merge'] != 'both').sum())} new rows without an old row "
                         f"({len(new)} new, {len(old)} old): stop and paste this")
    worst, bad = 0.0, []
    for c in sorted((set(old.columns) & set(new.columns)) - set(keys)):
        x, y = m[f"{c}_n"].to_numpy(), m[f"{c}_o"].to_numpy()
        if np.issubdtype(x.dtype, np.floating) and np.issubdtype(y.dtype, np.floating):
            d = float(np.nanmax(np.abs(x - y))) if len(x) else 0.0
            worst = max(worst, d)
            if d > tol or not np.array_equal(np.isnan(x), np.isnan(y)):
                bad.append(f"{c} ({d:.1e})")
        elif not np.array_equal(x.astype(str), y.astype(str)):
            bad.append(f"{c} ({int((x.astype(str) != y.astype(str)).sum())} rows)")
    if bad:
        raise SystemExit(f"{label}: shared rows differ in {bad}: stop and paste this")
    conds = sorted(new.condition.unique())
    print(f"check ({label}): {len(new)} shared rows ({', '.join(conds)}) identical to the stored bench "
          f"(largest float difference {worst:.1e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--by", default="", help="extra key columns, e.g. method")
    a = ap.parse_args()
    check_subset(pd.read_parquet(a.old), pd.read_parquet(a.new), [c for c in a.by.split(",") if c], a.new)


if __name__ == "__main__":
    sys.exit(main())
