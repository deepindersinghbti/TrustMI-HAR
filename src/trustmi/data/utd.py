"""UTD-MHAD loader with per-fold, train-only normalisation (blueprint D-6)."""
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
TENSORS = ROOT / "data/processed/utd_mhad_v1.npz"
VIDEO = dict(v1=ROOT / "data/features/video_r2plus1d18_v1.npz",   # conference: centre crop 112x112
             v2=ROOT / "data/features/video_r2plus1d18_v2.npz")   # journal: whole frame (docs/data_audit_mmact_video.md)
SPLITS = ROOT / "splits/utd_mhad_loso_v1.json"
EPS = 1e-6


def _load_all(video="v1"):
    m = np.load(TENSORS, allow_pickle=True)
    v = np.load(VIDEO[video], allow_pickle=True)
    split = json.loads(SPLITS.read_text())
    trials = m["trial"]
    if not (trials == v["trial"]).all():
        raise RuntimeError("video features and tensors are in different trial order")
    h = hashlib.sha256("|".join(trials.tolist()).encode()).hexdigest()[:16]
    if h != split["trial_order_sha"]:
        raise RuntimeError(f"split file was made for a different trial order ({h})")
    data = dict(imu=m["X_imu"].astype(np.float64),       # (N, 100, 6)
                skel=m["X_skel"].astype(np.float64),     # (N, 40, 20, 3)
                video=v["tokens"].astype(np.float64),    # (N, 4, 512)
                y=m["y"].astype(np.int64), subject=m["subject"],
                group=m["group"], trial=trials)
    return data, split


def fit_stats(data, train_idx):
    """Mean/std for each modality, computed ONLY from training rows."""
    stats = {}
    for k, axes in [("imu", (0, 1)), ("skel", (0, 1)), ("video", (0, 1))]:
        x = data[k][train_idx]
        stats[k] = (x.mean(axis=axes), x.std(axis=axes) + EPS)
    return stats


class UTDMHAD(Dataset):
    def __init__(self, fold, part, stats_dir=ROOT / "data/processed/norm_stats", video="v1"):
        assert part in ("train", "val", "test")
        data, split = _load_all(video)
        f = split["folds"][fold]
        self.fold, self.part, self.video = fold, part, video
        self.idx = np.array(f[part])
        train_idx = np.array(f["train"])
        stats = fit_stats(data, train_idx)

        stats_dir = Path(stats_dir); stats_dir.mkdir(parents=True, exist_ok=True)
        tag = "utd_v1" if video == "v1" else f"utd_v1_video{video}"     # v1 file name unchanged
        self.stats_file = stats_dir / f"{tag}_fold{fold}.npz"            # read by trustmi.operators
        np.savez(self.stats_file,
                 **{f"{k}_{s}": v[i] for k, v in stats.items()
                    for i, s in enumerate(("mean", "std"))},
                 train_subjects=np.array(f["train_subjects"]))

        self.x = {k: torch.from_numpy(((data[k][self.idx] - mu) / sd).astype(np.float32))
                  for k, (mu, sd) in stats.items()}
        self.y = torch.from_numpy(data["y"][self.idx])
        self.subject = data["subject"][self.idx]
        self.group = data["group"][self.idx]
        self.trial = data["trial"][self.idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return dict(imu=self.x["imu"][i], skel=self.x["skel"][i],
                    video=self.x["video"][i], y=self.y[i], index=int(self.idx[i]))
