"""WearGait-PD plan, step 1 (docs/plan_weargait_2026-11-11.md): recordings -> 2 s clips.

  python scripts/build_weargait.py                     all 1,473 recordings (a few minutes)
  python scripts/build_weargait.py --limit 3           first 3 recordings per group (code check)

Per recording (CSV, 100 Hz):
  1. 13 IMUs, accelerometer + gyroscope (6 channels each), grouped into 3 sensors (plan 2.3, 2.4):
       arms  = L_Wrist, R_Wrist                                                         12 channels
       trunk = Forehead, Xiphoid, LowerBack                                             18 channels
       legs  = L/R MidLatThigh, L/R LatShank, L/R Ankle, L/R DorsalFoot                 48 channels
     An IMU that is absent from a recording is all missing (NaN).
  2. Anti-aliasing low-pass (31-tap Hamming-windowed sinc, cut-off 20 Hz, zero phase), then every 2nd sample
     -> 50 Hz. Missing samples are bridged by linear interpolation ONLY for the filter and set back to NaN after.
  3. Labels (GeneralEvent): the six balance stances (EO_*, EC_*) -> Standing; SitToStand and unlabeled dropped
     (plan 2.1). Segments = runs of one label in the 100 Hz recording.
  4. Clips: 100 samples (2 s at 50 Hz), step 50 (1 s), fully inside one segment, at most 8 per segment, taken
     from the start of the segment (plan 2.2).
  5. A clip is CLEAN if no sample of any channel within the clip, widened by the filter's reach (15 samples at
     100 Hz on each side), is missing. Otherwise it is a REAL-DROPOUT clip; the sensors with missing samples
     are recorded (plan 2.6).
Writes data/processed/weargait_v1.npz (arrays) and weargait_v1_clips.parquet (one row per clip) and
weargait_v1.json (settings and counts). Deterministic: same input files -> byte-identical arrays.
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/WearGait-PD"
OUT = ROOT / "data/processed"
GROUPS = {"PD": RAW / "PD_PARTICIPANTS/CSV files", "control": RAW / "CONTROL_PARTICIPANTS/CSV files"}
SENSORS = {"arms": ["L_Wrist", "R_Wrist"],
           "trunk": ["Forehead", "Xiphoid", "LowerBack"],
           "legs": ["L_MidLatThigh", "R_MidLatThigh", "L_LatShank", "R_LatShank",
                    "L_Ankle", "R_Ankle", "L_DorsalFoot", "R_DorsalFoot"]}
AXES = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
CLASSES = ["Walk", "Standing", "TandemWalk", "Turn", "Sitting", "TurnToSit"]       # plan 2.1, fixed order
DROPPED = {"SitToStand", "unlabeled"}
FS_IN, DEC = 100, 2                     # 100 Hz in, keep every 2nd sample -> 50 Hz
T, STEP, CAP = 100, 50, 8               # 2 s clips, 1 s step, at most 8 clips per segment
TAPS, CUTOFF = 31, 20.0                 # low-pass filter
REACH = TAPS // 2                       # samples (100 Hz) a missing value can influence through the filter


def channels(sensor):
    return [f"{loc}_{a}" for loc in SENSORS[sensor] for a in AXES]


def lowpass_kernel():
    n = np.arange(TAPS) - REACH
    h = np.sinc(2 * CUTOFF / FS_IN * n) * np.hamming(TAPS)
    return h / h.sum()


KERNEL = lowpass_kernel()


def label_class(lab):
    if re.match(r"^E[OC]_", lab):
        return "Standing"
    return None if lab in DROPPED else lab


def filter_and_decimate(x):
    """x (n, C) at 100 Hz with NaN -> (values (m, C) at 50 Hz, NaN where the sample was missing)."""
    out = np.empty_like(x)
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        v = x[:, c]
        ok = np.isfinite(v)
        if not ok.any():
            out[:, c] = np.nan
            continue
        filled = np.interp(idx, idx[ok], v[ok]) if not ok.all() else v
        padded = np.pad(filled, REACH, mode="edge")              # edges: hold the first / last value
        out[:, c] = np.convolve(padded, KERNEL, mode="valid")
        out[~ok, c] = np.nan                                      # bridged samples stay missing
    return out[::DEC]


def scan(path, group):
    person, task = path.stem.split("_", 1)
    want = ["Time", "GeneralEvent"] + [c for s in SENSORS for c in channels(s)]
    d = pd.read_csv(path, low_memory=False, usecols=lambda c: c in want)
    t = d["Time"].astype(str).str.replace(" sec", "", regex=False).astype(float).to_numpy()
    dt = np.diff(t)
    n_jumps = int((np.abs(dt - 1 / FS_IN) > 0.1 / FS_IN).sum())  # steps off by more than 10%
    raw = {s: np.stack([pd.to_numeric(d[c], errors="coerce").to_numpy(np.float64) if c in d
                        else np.full(len(d), np.nan) for c in channels(s)], axis=1) for s in SENSORS}
    absent = {s: [loc for loc in SENSORS[s] if f"{loc}_Acc_X" not in d] for s in SENSORS}
    miss100 = {s: ~np.isfinite(raw[s]).all(axis=1) for s in SENSORS}
    # widened by the filter's reach: a clean clip is untouched by any bridged value
    wide = {s: np.convolve(m.astype(int), np.ones(2 * REACH + 1, int), mode="same") > 0 for s, m in miss100.items()}
    sig = {s: filter_and_decimate(raw[s]) for s in SENSORS}
    wide50 = {s: (np.r_[w, w[-1:]] if len(w) % DEC else w).reshape(-1, DEC).any(1) for s, w in wide.items()}
    m50 = len(sig["arms"])

    lab = d["GeneralEvent"].astype(str).to_numpy()
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    clips = []
    for k, (a, b) in enumerate(zip(np.r_[0, ch], np.r_[ch, len(lab)])):
        cls = label_class(lab[a])
        if cls is None:
            continue
        a50, b50 = -(-a // DEC), b // DEC                         # 50 Hz samples fully inside the segment
        b50 = min(b50, m50)
        for j in range(CAP):
            s0 = a50 + j * STEP
            if s0 + T > b50:
                break
            bad = [s for s in SENSORS if wide50[s][s0: s0 + T].any()]
            clips.append(dict(group=group, person=person, task=task, file=path.name, segment=k, label=lab[a],
                              cls=cls, y=CLASSES.index(cls), start=s0, clean=not bad,
                              missing=",".join(bad),
                              **{f"miss_{s}": float(np.isnan(sig[s][s0: s0 + T]).any(1).mean()) for s in SENSORS}))
    X = {s: (np.stack([sig[s][c["start"]: c["start"] + T] for c in clips]).astype(np.float32) if clips
             else np.zeros((0, T, len(channels(s))), np.float32)) for s in SENSORS}
    info = dict(group=group, person=person, task=task, file=path.name, n100=len(d), n_time_jumps=n_jumps,
                absent=";".join(f"{s}:{'|'.join(v)}" for s, v in absent.items() if v), n_clips=len(clips))
    return clips, X, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="weargait_v1")
    args = ap.parse_args()
    t0, rows, files, Xs = time.time(), [], [], {s: [] for s in SENSORS}
    for group, folder in GROUPS.items():
        paths = sorted(p for p in folder.glob("*.csv") if p.name != "manifest.csv")[: args.limit]
        for i, p in enumerate(paths):
            clips, X, info = scan(p, group)
            rows += clips; files.append(info)
            for s in SENSORS:
                Xs[s].append(X[s])
            if (i + 1) % 100 == 0:
                print(f"{group}: {i + 1}/{len(paths)} ({time.time() - t0:.0f} s)", flush=True)
    C = pd.DataFrame(rows)
    C.insert(0, "clip", np.arange(len(C)))
    X = {s: np.concatenate(Xs[s]) for s in SENSORS}
    for s in SENSORS:
        if X[s].shape[0] != len(C):
            raise SystemExit(f"{s}: {X[s].shape[0]} arrays for {len(C)} clips")
        clean_nan = np.isnan(X[s][C.clean.to_numpy()]).any()
        if clean_nan:
            raise SystemExit(f"{s}: a CLEAN clip contains a missing value")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{args.out}.npz", **X, y=C.y.to_numpy(np.int64), clean=C.clean.to_numpy(),
             person=C.person.to_numpy(str), classes=np.array(CLASSES))
    C.to_parquet(OUT / f"{args.out}_clips.parquet", index=False)
    Fi = pd.DataFrame(files)
    Fi.to_parquet(OUT / f"{args.out}_files.parquet", index=False)
    summary = dict(settings=dict(sensors=SENSORS, axes=AXES, classes=CLASSES, dropped=sorted(DROPPED),
                                 fs_in=FS_IN, fs_out=FS_IN // DEC, clip_samples=T, step=STEP, cap=CAP,
                                 lowpass=dict(taps=TAPS, cutoff_hz=CUTOFF, window="hamming"), reach=REACH),
                   n_files=len(Fi), n_clips=len(C), n_clean=int(C.clean.sum()),
                   n_dropout=int((~C.clean).sum()), files_with_time_jumps=int((Fi.n_time_jumps > 0).sum()))
    (OUT / f"{args.out}.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print(f"\n{len(Fi)} recordings -> {len(C)} clips ({C.clean.sum()} clean, {(~C.clean).sum()} real-dropout) "
          f"in {time.time() - t0:.0f} s")
    print("clips per class (clean):"); print(C[C.clean].groupby(["cls", "group"]).size().unstack().fillna(0).astype(int))
    print("people with clean clips:", C[C.clean].groupby("group").person.nunique().to_dict())
    print("real-dropout clips by missing sensors:", (~C.clean).groupby(C.missing).sum().to_dict())
    print("recordings with time-step jumps:", int((Fi.n_time_jumps > 0).sum()),
          "| with an absent IMU:", int((Fi.absent != "").sum()))


if __name__ == "__main__":
    main()
