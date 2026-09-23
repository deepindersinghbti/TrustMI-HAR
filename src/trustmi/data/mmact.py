"""MMAct V1 sensor loader: 15 s windows at 50 Hz, per-fold train-only normalisation.

Modalities
  watch: watch accelerometer (3 ch, m/s^2)
  phone: phone accelerometer (3) + gyroscope (3) + sin/cos of pitch and roll (4) = 10 ch
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
SPLITS = ROOT / "splits/mmact_loso_v1.json"
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


def fit_stats(x, lens, rows):
    """Per-channel mean/std over the VALID samples of training rows only (padding excluded)."""
    valid = np.arange(T)[None, :] < lens[rows, None]
    v = x[rows][valid]
    return v.mean(0, dtype=np.float64), v.std(0, dtype=np.float64) + EPS


class MMAct(Dataset):
    def __init__(self, fold, part, stats_dir=ROOT / "data/processed/norm_stats"):
        assert part in ("train", "val", "test")
        x_all, lens, meta, split = _windows()
        f = split["folds"][fold]
        self.fold, self.part = fold, part
        self.idx = np.flatnonzero(np.isin(meta["subject"], f[part]))
        train_idx = np.flatnonzero(np.isin(meta["subject"], f["train"]))
        stats = {k: fit_stats(v, lens, train_idx) for k, v in x_all.items()}

        stats_dir = Path(stats_dir); stats_dir.mkdir(parents=True, exist_ok=True)
        np.savez(stats_dir / f"mmact_v1_fold{fold}.npz",
                 **{f"{k}_{s}": v[i] for k, v in stats.items() for i, s in enumerate(("mean", "std"))},
                 train_subjects=np.array(f["train"]))

        valid = np.arange(T)[None, :, None] < lens[self.idx, None, None]
        self.x = {k: torch.from_numpy(np.where(valid, (x_all[k][self.idx] - mu) / sd, 0).astype(np.float32))
                  for k, (mu, sd) in stats.items()}
        self.x["len"] = torch.from_numpy(lens[self.idx])
        self.y = torch.from_numpy(meta["y"][self.idx])
        self.subject = meta["subject"][self.idx]
        self.group = meta["scene"][self.idx]    # scene = MMAct's confound (as placement group is for UTD)
        self.trial = meta["trial"][self.idx]
        self.classes = meta["classes"]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return dict(watch=self.x["watch"][i], phone=self.x["phone"][i], len=self.x["len"][i],
                    y=self.y[i], index=int(self.idx[i]))
