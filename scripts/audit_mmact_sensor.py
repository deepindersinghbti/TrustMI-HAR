"""Audit MMAct V1 trimmed sensor data: one row per (stream, clip) CSV file."""
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path("data/extracted/MMAct/sensor/trimmed")
OUT = Path("data/processed/mmact_sensor_index.parquet")
STREAMS = ["acc_phone_clip", "acc_watch_clip", "gyro_clip", "orientation_clip"]
KEY = ["subject", "scene", "session", "action"]
PAT = re.compile(r"subject(\d+)/scene(\d+)/session(\d+)/(.+)\.csv")


def inspect(path):
    rel = path.relative_to(ROOT).as_posix()
    stream, rest = rel.split("/", 1)
    m = PAT.fullmatch(rest)
    row = dict(stream=stream, file=rel, parse_ok=m is not None, error=None)
    if m:
        row.update(subject=int(m[1]), scene=int(m[2]), session=int(m[3]), action=m[4])
    try:
        d = pd.read_csv(path, header=None)
        row["n_rows"], row["n_cols"] = d.shape
        t = pd.to_datetime(d[0], format="%Y%m%d_%H:%M:%S.%f", errors="coerce")
        row["bad_ts"] = int(t.isna().sum())
        t = t.dropna()
        if len(t) > 1:
            dt = t.diff().dt.total_seconds().iloc[1:]
            row.update(t_start=t.iloc[0], t_end=t.iloc[-1],
                       duration_s=(t.iloc[-1] - t.iloc[0]).total_seconds(),
                       fs_median=1 / dt.median() if dt.median() > 0 else np.nan,
                       nonmono=int((dt <= 0).sum()), max_gap_s=dt.max())
        v = d.iloc[:, 1:].to_numpy(dtype=float)
        row.update(vmin=np.nanmin(v), vmax=np.nanmax(v), n_nan=int(np.isnan(v).sum()))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


if __name__ == "__main__":
    files = sorted(p for s in STREAMS for p in (ROOT / s).rglob("*.csv"))
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame(ex.map(inspect, files, chunksize=64))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"files {len(df)} | path-parse failures {(~df.parse_ok).sum()} | "
          f"read errors {df.error.notna().sum()}")
    print(df[df.error.notna()][["file", "error"]].head(10).to_string())
    print("\ncolumns per file:", df.n_cols.value_counts(dropna=False).to_dict())

    ok = df[df.parse_ok & df.error.isna()]
    print("subjects:", sorted(ok.subject.unique()))
    print("scenes:", sorted(ok.scene.unique()), "| sessions:", sorted(ok.session.unique()))

    print("\nPER-STREAM SUMMARY")
    print(ok.groupby("stream").agg(
        files=("file", "size"), fs_hz=("fs_median", "median"),
        dur_med=("duration_s", "median"), dur_min=("duration_s", "min"),
        dur_max=("duration_s", "max"), rows_min=("n_rows", "min"),
        vmin=("vmin", "min"), vmax=("vmax", "max"),
        nonmono_files=("nonmono", lambda s: int((s > 0).sum())),
        max_gap_s=("max_gap_s", "max"), bad_ts=("bad_ts", "sum"),
        nan=("n_nan", "sum")).round(2).T.to_string())

    acts = ok[ok.stream == "acc_phone_clip"].action.value_counts().sort_index()
    print(f"\nACTIONS ({ok.action.nunique()} distinct names across all streams)")
    print(acts.to_string())
    print("names with 'transfer' or 'loiter':",
          sorted(n for n in ok.action.unique() if "transf" in n or "loiter" in n))

    cov = (ok.assign(v=1).pivot_table(index=KEY, columns="stream", values="v",
                                      aggfunc="sum", fill_value=0))
    print(f"\nCOVERAGE: {len(cov)} unique clips | duplicate files {int((cov > 1).sum().sum())} | "
          f"all four streams {int((cov > 0).all(axis=1).sum())}")
    print("stream presence patterns (1 = present):")
    print((cov > 0).astype(int).value_counts().to_string())
    nw = cov[cov["acc_watch_clip"] == 0].reset_index()
    print("clips missing watch, by subject:", nw.subject.value_counts().sort_index().to_dict())

    one = ok.drop_duplicates(KEY + ["stream"])
    ts = one.pivot(index=KEY, columns="stream", values="t_start").dropna()
    te = one.pivot(index=KEY, columns="stream", values="t_end").dropna()
    s_spread = (ts.max(axis=1) - ts.min(axis=1)).dt.total_seconds()
    e_spread = (te.max(axis=1) - te.min(axis=1)).dt.total_seconds()
    print(f"\nCROSS-STREAM ALIGNMENT over {len(ts)} clips with all four streams (seconds)")
    print("start spread:", s_spread.describe(percentiles=[.5, .9, .99]).round(3).to_dict())
    print("end spread:  ", e_spread.describe(percentiles=[.5, .9, .99]).round(3).to_dict())
