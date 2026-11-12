"""WearGait-PD loader (docs/plan_weargait_2026-11-11.md): 2 s clips at 50 Hz, 3 sensors, per-fold
train-only normalisation.

Sensors (accelerometer + gyroscope of each IMU, m/s^2 and rad/s; built by scripts/build_weargait.py)
  arms   L/R wrist                                                       12 channels
  trunk  forehead, chest (xiphoid), lower back                           18 channels
  legs   L/R thigh, shank, ankle, foot                                   48 channels
Parts: train / val / test = CLEAN clips of the fold's people; 'dropout' = the REAL-DROPOUT clips of the fold's
TEST people (never used for training or fitting). In those, missing samples are set to the training mean
(0 after normalisation): the r_fill rule. Same attributes as MMAct (x, y, subject, group, trial, idx, classes,
stats_file); group = cohort ('PD' / 'control'). Every clip is full length (len = 100).
"""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/processed/weargait_v1.npz"
CLIPS = ROOT / "data/processed/weargait_v1_clips.parquet"
SPLITS = ROOT / "splits/weargait_v1.json"
SENSORS = ("arms", "trunk", "legs")
T, EPS = 100, 1e-6
PARTS = ("train", "val", "test", "dropout")
_CACHE = {}


def _load():
    if "d" not in _CACHE:
        with np.load(DATA) as d:
            arrays = {k: d[k] for k in (*SENSORS, "y", "clean", "person", "classes")}
        C = pd.read_parquet(CLIPS)
        split = json.loads(SPLITS.read_text())
        h = hashlib.sha256("\n".join(C.file + "#" + C.segment.astype(str) + "@" + C.start.astype(str))
                           .encode()).hexdigest()[:16]
        if h != split["clip_order_hash"]:
            raise RuntimeError(f"split file was made for a different clip order ({h})")
        if not (np.array_equal(arrays["person"], C.person.to_numpy(str))
                and np.array_equal(arrays["clean"], C.clean.to_numpy())):
            raise RuntimeError("weargait_v1.npz and weargait_v1_clips.parquet do not match")
        _CACHE["d"] = (arrays, C, split)
    return _CACHE["d"]


def fit_stats(x, rows):
    """Per-channel mean / std over the given (clean) clips and all their samples."""
    v = x[rows]
    return v.mean((0, 1), dtype=np.float64), v.std((0, 1), dtype=np.float64) + EPS


class WearGait:
    def __init__(self, fold, part, stats_dir=ROOT / "data/processed/norm_stats"):
        if part not in PARTS:
            raise ValueError(part)
        arrays, C, split = _load()
        f = split["folds"][fold]
        person, clean = arrays["person"], arrays["clean"]
        train_idx = np.flatnonzero(clean & np.isin(person, f["train"]))
        people = f["test"] if part == "dropout" else f[part]
        self.idx = np.flatnonzero(np.isin(person, people) & (~clean if part == "dropout" else clean))
        stats = {s: fit_stats(arrays[s], train_idx) for s in SENSORS}
        self.stats = stats
        if part != "dropout":                    # as MMAct: every main-part loader (re)writes the same file
            stats_dir = Path(stats_dir); stats_dir.mkdir(parents=True, exist_ok=True)
            self.stats_file = stats_dir / f"weargait_v1_fold{fold}.npz"      # read by trustmi.operators
            np.savez(self.stats_file,
                     **{f"{k}_{s}": v[i] for k, v in stats.items() for i, s in enumerate(("mean", "std"))},
                     # the people the statistics come from (4 people have no clean clip, so they are absent
                     # from the training data; trustmi.operators checks this list against the training split)
                     train_subjects=np.unique(person[train_idx]))
        else:
            self.stats_file = None
        self.x = {}
        for s in SENSORS:
            z = (arrays[s][self.idx] - stats[s][0]) / stats[s][1]
            self.x[s] = torch.from_numpy(np.nan_to_num(z, nan=0.0).astype(np.float32))   # NaN only in 'dropout'
        self.x["len"] = torch.full((len(self.idx),), T, dtype=torch.int64)
        rows = C.iloc[self.idx]
        self.y = torch.from_numpy(arrays["y"][self.idx].astype(np.int64))
        self.subject = person[self.idx]
        self.group = rows.group.to_numpy()
        self.trial = (rows.file + "#" + rows.segment.astype(str) + "@" + rows.start.astype(str)).to_numpy()
        self.task = rows.task.to_numpy()
        self.missing = rows.missing.to_numpy()   # sensors with missing samples ('' for clean clips)
        # labelled segment of each clip: clips of one segment overlap in time, so they are never used as
        # swap donors or MI wrong partners for each other (trustmi.operators, scripts/train_mi.py)
        self.segment = (rows.file + "#" + rows.segment.astype(str)).to_numpy()
        self.classes = arrays["classes"]
        self.fold, self.part = fold, part

    def __len__(self):
        return len(self.idx)
