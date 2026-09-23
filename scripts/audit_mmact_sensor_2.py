"""MMAct sensor audit, pass 2: timestamp glitches, sampling rate, saturation, label/scene structure."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path("data/extracted/MMAct/sensor/trimmed")
FS = {"acc_phone_clip": 100, "acc_watch_clip": 100, "gyro_clip": 50, "orientation_clip": 50}
KEY = ["subject", "scene", "session", "action"]


def parse_ts(raw):
    """Normal rows have a fractional second; whole-second rows do not."""
    t = pd.to_datetime(raw, format="%Y%m%d_%H:%M:%S.%f", errors="coerce")
    miss = t.isna()
    t[miss] = pd.to_datetime(raw[miss], format="%Y%m%d_%H:%M:%S", errors="coerce")
    return t, miss


def inspect(args):
    rel, fs, peak = args
    d = pd.read_csv(ROOT / rel, header=None)
    raw = d[0].astype(str)
    t, miss = parse_ts(raw)
    dt = t.diff().dt.total_seconds().to_numpy()[1:]
    near = lambda x: np.abs(dt - x) < 0.5
    known = near(60) | near(-60) | near(3600) | near(-3600)
    tt = t.dropna()
    dur = (tt.iloc[-1] - tt.iloc[0]).total_seconds() if len(tt) > 1 else np.nan
    v = np.abs(d.iloc[:, 1:].to_numpy(float))
    return dict(
        file=rel,
        whole_sec_example=raw[miss].iloc[0] if miss.any() else "",
        unparsed=int(t.isna().sum()),
        dup=int((dt == 0).sum()),
        back_small=int(((dt < 0) & (dt > -0.5)).sum()),
        back_60=int(near(-60).sum()), fwd_60=int(near(60).sum()),
        back_3600=int(near(-3600).sum()), fwd_3600=int(near(3600).sum()),
        other_big=int(((np.abs(dt) >= 0.5) & ~known).sum()),
        rate_ratio=(len(d) - 1) / (fs * dur) if dur > 0 else np.nan,
        sat_frac=float((v >= 0.98 * peak).mean()),
    )


if __name__ == "__main__":
    idx = pd.read_parquet("data/processed/mmact_sensor_index.parquet")
    ok = idx[idx.error.isna()].copy()
    peak = {s: max(abs(g.vmin.min()), abs(g.vmax.max())) for s, g in ok.groupby("stream")}
    jobs = [(f, FS[s], peak[s]) for f, s in zip(ok.file, ok.stream)]
    with ProcessPoolExecutor() as ex:
        r = pd.DataFrame(ex.map(inspect, jobs, chunksize=64))
    r = r.merge(ok[["file", "stream"] + KEY], on="file")
    r.to_parquet("data/processed/mmact_sensor_audit2.parquet", index=False)

    print("1) WHOLE-SECOND ROWS")
    print("   examples:", r.loc[r.whole_sec_example != "", "whole_sec_example"].head(3).tolist())
    print("   rows still unparsed after the second format:", int(r.unparsed.sum()))

    print("\n2) TIMESTAMP STEPS  (files affected / total events)")
    cols = ["dup", "back_small", "back_60", "fwd_60", "back_3600", "fwd_3600", "other_big"]
    print(pd.DataFrame({c: r.groupby("stream")[c].agg(lambda s: f"{(s > 0).sum()} / {s.sum()}")
                        for c in cols}).to_string())

    print("\n3) ROWS PER SECOND vs NOMINAL RATE  (1.00 = exactly 100 or 50 Hz)")
    print(r.groupby("stream").rate_ratio.describe(percentiles=[.01, .5, .99]).round(3).to_string())

    print("\n4) SATURATION  (samples within 2% of the stream's largest value)")
    for s in ["acc_phone_clip", "acc_watch_clip", "gyro_clip"]:
        g = r[r.stream == s]
        print(f"   {s:16s} peak {peak[s]:6.2f} | files touching it {int((g.sat_frac > 0).sum()):5d}"
              f" | worst file {g.sat_frac.max():.3f} of its samples")

    for c, target in [("back_60", -60), ("back_3600", -3600), ("dup", 0)]:
        hit = r[r[c] > 0]
        if hit.empty:
            continue
        f = hit.file.iloc[0]
        d = pd.read_csv(ROOT / f, header=None)
        t, _ = parse_ts(d[0].astype(str))
        dt = t.diff().dt.total_seconds().fillna(1e9).to_numpy()
        i = int(np.argmin(np.abs(dt - target)))
        print(f"\n5) EXAMPLE {c}: {f}, rows {max(i - 3, 0)}-{i + 2}")
        print(d.iloc[max(i - 3, 0): i + 3, :2].to_string(header=False))

    lab = ok.assign(action=ok.action.str.lower())
    clips = lab.drop_duplicates(KEY)
    print("\n6) CLIPS PER ACTION x SCENE")
    print(pd.crosstab(clips.action, clips.scene, margins=True).to_string())

    cov = lab.pivot_table(index=KEY, columns="stream", values="file", aggfunc="count", fill_value=0)
    nw = cov[cov["acc_watch_clip"] == 0].reset_index()
    print("\n7) CLIPS MISSING WATCH: subject x scene")
    print(pd.crosstab(nw.subject, nw.scene, margins=True).to_string())
    print("\n8) COMPLETE CLIPS PER SUBJECT:",
          cov[(cov > 0).all(axis=1)].reset_index().subject.value_counts().sort_index().to_dict())

    ph = lab[lab.stream == "acc_phone_clip"]
    long = ph[ph.duration_s > 20]
    print(f"\n9) CLIPS LONGER THAN 20 s: {len(long)} | by action:",
          long.action.value_counts().head(8).to_dict())
