"""Part C.3 of docs/plan_followup_2026-11-12.md: compare re-generated outputs with the frozen ones.

  python scripts/compare_outputs.py --frozen <project> --rerun <clone> --files a.csv b.parquet ...
  python scripts/compare_outputs.py --frozen <project> --rerun <clone> --logs frozen.log:rerun.log ...

Tables (.csv, .parquet): PASS if byte-identical, or same rows, columns and text values with every number within
1e-12 (absolute) of the frozen one; otherwise FAIL with the largest difference. Logs: lines compared as text;
any differing line is printed as a NOTE (a log is a printout; it decides nothing). Exit 1 if any table differs or
is missing.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

TOL = 1e-12


def read(p):
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def compare_table(a, b):
    if not b.exists():
        return False, "MISSING in the re-run"
    if a.read_bytes() == b.read_bytes():
        return True, "byte-identical"
    x, y = read(a), read(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return False, f"different shape/columns: {x.shape} vs {y.shape}"
    worst, where = 0.0, ""
    for c in x.columns:
        u, v = x[c], y[c]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
            u, v = u.to_numpy(float), v.to_numpy(float)
            if not np.array_equal(np.isnan(u), np.isnan(v)):
                return False, f"column {c}: missing values in different places"
            m = np.nan_to_num(np.abs(u - v), nan=0.0).max() if len(u) else 0.0
            if m > worst:
                worst, where = m, c
        else:
            if not u.astype(str).equals(v.astype(str)):
                k = int((u.astype(str) != v.astype(str)).to_numpy().argmax())
                return False, f"column {c}: text differs (row {k}: {u.iloc[k]!r} vs {v.iloc[k]!r})"
    if worst <= TOL:
        return True, f"equal within {TOL:g} (largest difference {worst:.1e}{', ' + where if where else ''})"
    return False, f"largest difference {worst:.3e} in column {where}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True); ap.add_argument("--rerun", required=True)
    ap.add_argument("--files", nargs="*", default=[]); ap.add_argument("--logs", nargs="*", default=[])
    a = ap.parse_args()
    F, R = Path(a.frozen), Path(a.rerun)
    bad = notes = 0
    for rel in a.files:
        ok, msg = compare_table(F / rel, R / rel)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {rel}: {msg}")
    for pair in a.logs:
        fl, rl = pair.split(":")
        x = (F / fl).read_text().splitlines() if (F / fl).exists() else None
        y = (R / rl).read_text().splitlines() if (R / rl).exists() else None
        if x is None or y is None:
            print(f"NOTE  log {fl}: {'frozen' if x is None else 're-run'} log missing"); notes += 1; continue
        diff = [(i, p, q) for i, (p, q) in enumerate(zip(x, y)) if p != q]
        if len(x) != len(y):
            diff.append((min(len(x), len(y)), f"<{len(x)} lines>", f"<{len(y)} lines>"))
        notes += bool(diff)
        print(f"{'same' if not diff else 'NOTE'}  log {fl} vs {rl}: "
              + ("identical" if not diff else f"{len(diff)} line(s) differ"))
        for i, p, q in diff[:5]:
            print(f"        line {i + 1}: frozen | {p}\n                 re-run | {q}")
    print(f"\n{len(a.files)} tables compared: {bad} differ | {len(a.logs)} logs compared: {notes} with differing lines")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
