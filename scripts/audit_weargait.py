"""Data audit of WearGait-PD (docs/data_audit_weargait.md). READS ONLY, trains nothing.

  python scripts/audit_weargait.py                 all files (~10 min: 1,473 CSV files, 10-60 MB each)
  python scripts/audit_weargait.py --limit 20      first 20 files per group (quick look)

Per file it records: person, group (PD / control), task, duration, sampling rate, which of the 13 IMU
locations are present and how much of each is missing (NaN), and every labelled segment of the GeneralEvent
column (label, start, length). Writes two tables and prints a summary that answers the audit questions:
  data/processed/weargait_files.parquet      one row per file
  data/processed/weargait_segments.parquet   one row per labelled segment
"""
import argparse, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/WearGait-PD"
GROUPS = {"PD": RAW / "PD_PARTICIPANTS/CSV files", "control": RAW / "CONTROL_PARTICIPANTS/CSV files"}
IMU = re.compile(r"^(.+?)_(Acc|Gyr|Mag|FreeAcc|VelInc|OriInc|Roll|Pitch|Yaw)")


def scan(path, group):
    name = path.stem
    person, task = name.split("_", 1)
    d = pd.read_csv(path, low_memory=False, usecols=lambda c: not c.startswith("Walkway"))
    t = d["Time"].astype(str).str.replace(" sec", "", regex=False).astype(float).to_numpy()
    rate = float(1 / np.median(np.diff(t))) if len(t) > 2 else np.nan
    locs = sorted({m.group(1) for c in d.columns for m in [IMU.match(c)] if m})
    row = dict(group=group, person=person, task=task, file=path.name, n=len(d), duration_s=float(t[-1] - t[0]),
               rate_hz=rate, n_imu=len(locs), imu=",".join(locs))
    for loc in locs:
        acc = d.get(f"{loc}_Acc_X")
        row[f"nan_{loc}"] = float(acc.isna().mean()) if acc is not None else np.nan
    g = d["GeneralEvent"].astype(str).to_numpy()
    ch = np.flatnonzero(g[1:] != g[:-1]) + 1
    starts, ends = np.r_[0, ch], np.r_[ch, len(g)]
    segs = [dict(group=group, person=person, task=task, label=g[s], start_s=float(t[s]),
                 length_s=float(t[e - 1] - t[s] + (t[1] - t[0]))) for s, e in zip(starts, ends)]
    row["clinical_labels"] = ",".join(sorted(set(d["ClinicalEvent"].astype(str)) - {"unlabeled", "nan"}))
    return row, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows, segs, t0 = [], [], time.time()
    for group, folder in GROUPS.items():
        files = sorted(p for p in folder.glob("*.csv") if p.name != "manifest.csv")[: args.limit]
        for i, p in enumerate(files):
            try:
                r, s = scan(p, group)
            except Exception as e:                       # record, do not stop
                r, s = dict(group=group, person=p.stem.split("_")[0], task=p.stem.split("_", 1)[1], file=p.name,
                            error=str(e)[:200]), []
            rows.append(r); segs += s
            if (i + 1) % 50 == 0:
                print(f"{group}: {i + 1}/{len(files)} files ({time.time() - t0:.0f} s)", flush=True)
    F, S = pd.DataFrame(rows), pd.DataFrame(segs)
    out = ROOT / "data/processed"; out.mkdir(parents=True, exist_ok=True)
    F.to_parquet(out / "weargait_files.parquet", index=False)
    S.to_parquet(out / "weargait_segments.parquet", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)

    print(f"\n=== FILES: {len(F)} ({F.groupby('group').person.nunique().to_dict()} people)")
    if "error" in F:
        bad = F[F.error.notna()]
        print(f"unreadable files: {len(bad)}"); print(bad[["file", "error"]].to_string() if len(bad) else "")
        F = F[F.error.isna()]
    print("files per task and group:"); print(F.pivot_table(index="task", columns="group", values="file", aggfunc="count"))
    print("\nsampling rate (Hz):", F.rate_hz.round(1).value_counts().to_dict())
    print("duration (s) per task:"); print(F.groupby("task").duration_s.describe()[["min", "50%", "max"]].round(1))
    print("\nIMU locations per file:", F.n_imu.value_counts().to_dict())
    print("location sets:", F.imu.value_counts().to_dict())
    nan_cols = [c for c in F.columns if c.startswith("nan_")]
    print("\nMISSING SENSOR DATA: share of files where a location is >1% NaN, and mean NaN share there")
    for c in nan_cols:
        v = F[c]; k = v > 0.01
        print(f"  {c[4:]:14s} affected files {k.mean():6.1%}   mean NaN share in them {v[k].mean() if k.any() else 0:.2f}"
              f"   fully missing (>99%) {(v > 0.99).mean():.1%}")
    print("files with ANY location >1% NaN:", f"{(F[nan_cols] > 0.01).any(axis=1).mean():.1%}")
    print("\nclinical event labels seen:", sorted(set(",".join(F.clinical_labels.dropna()).split(",")) - {""}))

    print(f"\n=== LABELLED SEGMENTS (GeneralEvent): {len(S)}")
    print("segments per label: count, median and 5% / 95% length (s), people with it")
    g = S.groupby("label")
    print(pd.DataFrame(dict(count=g.size(), median_s=g.length_s.median().round(1),
                            p5_s=g.length_s.quantile(.05).round(1), p95_s=g.length_s.quantile(.95).round(1),
                            people=g.person.nunique())).sort_values("count", ascending=False))
    print("\nsegments >= 2 s per label and group:")
    print(S[S.length_s >= 2].pivot_table(index="label", columns="group", values="person", aggfunc="count").fillna(0).astype(int))
    print("\nlabels per task:")
    print(S.groupby("task").label.unique().apply(lambda a: ", ".join(sorted(a))))


if __name__ == "__main__":
    main()
