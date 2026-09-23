"""Build MMAct V1 sensor tensors (v1).

Decisions (recorded in data/README.md):
- sample order = file order; per-row timestamps ignored (file order smoother in 97-100% of reordered files)
- common rate 50 Hz: accelerometers 100 -> 50 Hz with resample_poly (anti-aliased, line-padded edges
  so gravity does not dip at clip boundaries); gyro and orientation already 50 Hz, stored unchanged
- all streams start at t = 0 (start offsets median 10 ms, max 75 ms), trimmed to the shortest stream
- lossless ragged storage: clip i is X[start[i] : start[i] + length[i]]; windowing decided at load time
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path("data/extracted/MMAct/sensor/trimmed")
OUT = Path("data/processed/mmact_sensor_v1.npz")
FS_OUT = 50
STREAMS = [("acc_watch_clip", 100), ("acc_phone_clip", 100), ("gyro_clip", 50), ("orientation_clip", 50)]
CHANNELS = ["watch_acc_x", "watch_acc_y", "watch_acc_z", "phone_acc_x", "phone_acc_y", "phone_acc_z",
            "gyro_x", "gyro_y", "gyro_z", "orient_0", "orient_1", "orient_2"]
UNITS = ["m/s^2"] * 6 + ["rad/s"] * 3 + ["rad"] * 3
PEAK = {"acc_watch_clip": 78.38, "acc_phone_clip": 39.66, "gyro_clip": 17.46}  # observed maxima, audit pass 2
SAT_COL = {"acc_watch_clip": "sat_watch", "acc_phone_clip": "sat_phone", "gyro_clip": "sat_gyro"}
SPREAD_MAX, RATE_LO, RATE_HI = 1.10, 0.8, 1.25
KEY = ["subject", "scene", "session", "action"]


def load_clip(files):
    """files: {stream: relative path}. Returns (float32 array (T, 12) at 50 Hz, saturation fractions)."""
    parts, sat = {}, {}
    for s, fs in STREAMS:
        if s not in files:
            continue
        v = pd.read_csv(ROOT / files[s], header=None).iloc[:, 1:].to_numpy(np.float64)
        if s in PEAK:
            sat[s] = float((np.abs(v) >= 0.98 * PEAK[s]).mean())
        if fs != FS_OUT:
            v = resample_poly(v, 1, fs // FS_OUT, axis=0, padtype="line")
        parts[s] = v
    n = min(len(v) for v in parts.values())
    X = np.full((n, len(CHANNELS)), np.nan, np.float32)
    for k, (s, _) in enumerate(STREAMS):
        if s in parts:
            X[:, 3 * k: 3 * k + 3] = parts[s][:n]
    return X, sat


if __name__ == "__main__":
    idx = pd.read_parquet("data/processed/mmact_sensor_index.parquet")
    a2 = pd.read_parquet("data/processed/mmact_sensor_audit2.parquet")[["file", "rate_ratio"]]
    ok = idx[idx.error.isna()].merge(a2, on="file", how="left")
    ok["action"] = ok.action.str.lower()
    ok["dur_samples"] = ok.n_rows / ok.stream.map(dict(STREAMS))

    g = ok.groupby(KEY)
    clips = pd.DataFrame({
        "files": g[["stream", "file"]].apply(lambda d: dict(zip(d.stream, d.file))),
        "has_watch": g.stream.agg(lambda s: "acc_watch_clip" in set(s)),
        "n_streams": g.stream.nunique(),
        "spread": g.dur_samples.agg(lambda x: x.max() / x.min()),
        "rate_bad": g.rate_ratio.agg(lambda r: bool(((r < RATE_LO) | (r > RATE_HI)).any())),
    }).reset_index().sort_values(KEY).reset_index(drop=True)
    clips["status"] = np.select(
        [clips.n_streams < 3, clips.spread > SPREAD_MAX, clips.rate_bad, ~clips.has_watch],
        ["excluded_streams", "excluded_length", "excluded_rate", "no_watch"], default="main")

    with ProcessPoolExecutor() as ex:
        res = list(ex.map(load_clip, clips.files, chunksize=32))
    length = np.array([len(x) for x, _ in res])
    start = np.concatenate([[0], np.cumsum(length)[:-1]])
    X = np.concatenate([x for x, _ in res])
    for s, col in SAT_COL.items():
        clips[col] = [r[1].get(s, np.nan) for r in res]

    classes = sorted(clips.action.unique())
    y = clips.action.map({c: i for i, c in enumerate(classes)}).to_numpy()
    trial = [f"s{a:02d}_sc{b}_se{c}_{d}" for a, b, c, d in clips[KEY].itertuples(index=False)]
    np.savez(OUT, X=X, start=start, length=length, y=y, classes=np.array(classes),
             subject=clips.subject.to_numpy(), scene=clips.scene.to_numpy(),
             session=clips.session.to_numpy(), action=clips.action.to_numpy().astype(str),
             trial=np.array(trial), status=clips.status.to_numpy().astype(str),
             spread=clips.spread.to_numpy(), sat_watch=clips.sat_watch.to_numpy(),
             sat_phone=clips.sat_phone.to_numpy(), sat_gyro=clips.sat_gyro.to_numpy(),
             channels=np.array(CHANNELS), units=np.array(UNITS), fs=FS_OUT)
    clips.loc[clips.status.str.startswith("excluded"), KEY + ["status", "spread"]].to_csv(
        "data/processed/mmact_excluded_clips.csv", index=False)

    main = (clips.status == "main").to_numpy()
    mi = np.flatnonzero(main)
    seg = lambda i, sl=slice(None): X[start[i]: start[i] + length[i], sl]
    print(f"clips {len(clips)} | X {X.shape} {X.dtype} | {X.nbytes / 1e6:.0f} MB | classes {len(classes)}")
    print("status:", clips.status.value_counts().to_dict())
    print("main clips per subject:", clips[main].subject.value_counts().sort_index().to_dict())
    print("main clips per scene:", clips[main].scene.value_counts().sort_index().to_dict())
    print("main clips containing NaN:", sum(bool(np.isnan(seg(i)).any()) for i in mi))
    for name, sl in [("watch", slice(0, 3)), ("phone", slice(3, 6))]:
        m = [np.linalg.norm(seg(i, sl), axis=1).mean() for i in mi]
        print(f"median per-clip mean |acc| {name}: {np.median(m):.2f} m/s^2  (gravity = 9.81)")
    i = mi[0]
    raw = pd.read_csv(ROOT / clips.files[i]["gyro_clip"], header=None).iloc[:5, 1:].to_numpy(np.float32)
    print("gyro identity check (stored == raw CSV):", bool(np.allclose(raw, seg(i, slice(6, 9))[:5])))
    print("main duration at 50 Hz (s):",
          pd.Series(length[main] / FS_OUT).describe(percentiles=[.5, .95, .99]).round(2).to_dict())
    print("main clips with any saturated sample:",
          {c: int((clips.loc[main, c] > 0).sum()) for c in SAT_COL.values()})
    per_orient = [np.ptp(seg(i, slice(9, 12)), axis=0) for i in mi[:2000]]
    print("median within-clip range of orientation columns (rad):", np.median(per_orient, axis=0).round(2))
