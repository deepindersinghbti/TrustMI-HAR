"""MMAct V1 sensor loader: 15 s windows at 50 Hz, per-fold train-only normalisation.

Modalities
  watch: watch accelerometer (3 ch, m/s^2)
  phone: phone accelerometer (3) + gyroscope (3) + sin/cos of pitch and roll (4) = 10 ch
  video: (video=True only) frozen R(2+1)D-18 tokens (4 x 512), mean over the camera views, on the
         'main3' clips; see docs/data_audit_mmact_video.md
         (compass heading dropped: adds nothing to action accuracy and could carry room identity)
Clips longer than the window are centre-cropped; shorter ones are zero-padded AFTER normalisation,
and `len` (valid samples) is returned so the model can mask the padding.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
TENSORS = ROOT / "data/processed/mmact_sensor_v1.npz"
VIDEO = ROOT / "data/features/mmact_video_r2plus1d18_v2.npz"   # journal: video as 3rd sensor (main3 clips)
SPLITS = ROOT / "splits/mmact_loso_v1.json"
SENSORS = ("watch", "phone")
N_MAIN3 = 8432                                                   # docs/data_audit_mmact_video.md, decision 7
FS, WINDOW_S = 50, 15
T = FS * WINDOW_S
EPS = 1e-6
_CACHE = {}


def _windows():
    """All 'main' clips as fixed windows (not yet normalised) + metadata. Cached per process."""
    if "w" in _CACHE:
        return _CACHE["w"]
    d = np.load(TENSORS)
    split = json.loads(SPLITS.read_text())
    h = hashlib.sha256("\n".join(d["trial"]).encode()).hexdigest()[:16]
    if h != split["trial_order_hash"]:
        raise RuntimeError(f"split file was made for a different trial order ({h})")
    X, start, length = d["X"], d["start"], d["length"]
    keep = np.flatnonzero(d["status"] == "main")
    watch = np.zeros((len(keep), T, 3), np.float32)
    phone = np.zeros((len(keep), T, 10), np.float32)
    lens = np.zeros(len(keep), np.int64)
    for r, i in enumerate(keep):
        L = int(length[i])
        a, n = start[i] + max(0, (L - T) // 2), min(L, T)
        x = X[a: a + n]
        ang = x[:, 10:12]                                   # pitch, roll (orient_1, orient_2)
        watch[r, :n] = x[:, 0:3]
        phone[r, :n] = np.concatenate([x[:, 3:9], np.sin(ang), np.cos(ang)], axis=1)
        lens[r] = n
    meta = dict(y=d["y"][keep].astype(np.int64), subject=d["subject"][keep],
                scene=d["scene"][keep], trial=d["trial"][keep], classes=d["classes"])
    _CACHE["w"] = (dict(watch=watch, phone=phone), lens, meta, split)
    return _CACHE["w"]


def _with_video():
    """The 'main3' clips (main clips with usable video, decision 7) with video tokens added.
    Sensor windows are exactly those of _windows(); video rows are matched by trial NAME, never by position."""
    if "v" in _CACHE:
        return _CACHE["v"]
    x_all, lens, meta, split = _windows()
    with np.load(VIDEO) as f:                                              # unpack each array once
        v = {k: f[k] for k in ("trial", "status3", "tokens")}
    pos = {t: i for i, t in enumerate(v["trial"])}
    keep = np.array([r for r, t in enumerate(meta["trial"])
                     if t in pos and v["status3"][pos[t]] == "main3"])
    n_main3 = int((v["status3"] == "main3").sum())
    if len(keep) != n_main3 or n_main3 != N_MAIN3:
        raise RuntimeError(f"main3 mismatch: {len(keep)} sensor rows, {n_main3} in video file, expected {N_MAIN3}")
    rows = np.array([pos[t] for t in meta["trial"][keep]])
    x = {k: a[keep] for k, a in x_all.items()}
    x["video"] = v["tokens"][rows].astype(np.float32)                      # (N, 4, 512), mean over camera views
    meta3 = {k: (a if k == "classes" else a[keep]) for k, a in meta.items()}
    _CACHE["v"] = (x, lens[keep], meta3, split)
    return _CACHE["v"]


def fit_stats(x, lens, rows):
    """Per-channel mean/std over the VALID samples of training rows only (padding excluded)."""
    valid = np.arange(T)[None, :] < lens[rows, None]
    v = x[rows][valid]
    return v.mean(0, dtype=np.float64), v.std(0, dtype=np.float64) + EPS


class MMAct(Dataset):
    def __init__(self, fold, part, stats_dir=ROOT / "data/processed/norm_stats", video=False):
        assert part in ("train", "val", "test")
        x_all, lens, meta, split = _with_video() if video else _windows()
        f = split["folds"][fold]
        self.fold, self.part, self.video = fold, part, video
        self.idx = np.flatnonzero(np.isin(meta["subject"], f[part]))
        train_idx = np.flatnonzero(np.isin(meta["subject"], f["train"]))
        stats = {k: fit_stats(x_all[k], lens, train_idx) for k in SENSORS}
        if video:   # (N, 4, 512) tokens: per-dimension mean/std over training rows and tokens (as UTD-MHAD video)
            v = x_all["video"][train_idx]
            stats["video"] = (v.mean((0, 1), dtype=np.float64), v.std((0, 1), dtype=np.float64) + EPS)

        stats_dir = Path(stats_dir); stats_dir.mkdir(parents=True, exist_ok=True)
        tag = "mmact_v1_videov2" if video else "mmact_v1"                   # v1 file name unchanged
        self.stats_file = stats_dir / f"{tag}_fold{fold}.npz"            # read by trustmi.operators
        np.savez(self.stats_file,
                 **{f"{k}_{s}": v[i] for k, v in stats.items() for i, s in enumerate(("mean", "std"))},
                 train_subjects=np.array(f["train"]))

        valid = np.arange(T)[None, :, None] < lens[self.idx, None, None]
        self.x = {k: torch.from_numpy(np.where(valid, (x_all[k][self.idx] - stats[k][0]) / stats[k][1], 0)
                                      .astype(np.float32)) for k in SENSORS}
        if video:
            mu, sd = stats["video"]
            self.x["video"] = torch.from_numpy(((x_all["video"][self.idx] - mu) / sd).astype(np.float32))
        self.x["len"] = torch.from_numpy(lens[self.idx])
        self.y = torch.from_numpy(meta["y"][self.idx])
        self.subject = meta["subject"][self.idx]
        self.group = meta["scene"][self.idx]    # scene = MMAct's confound (as placement group is for UTD)
        self.trial = meta["trial"][self.idx]
        self.classes = meta["classes"]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.x.items()}, "y": self.y[i], "index": int(self.idx[i])}
