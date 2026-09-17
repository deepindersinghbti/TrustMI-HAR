import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

meta = np.load("data/processed/utd_mhad_v1.npz", allow_pickle=True)
vid = np.load("data/features/video_r2plus1d18_v1.npz", allow_pickle=True)
assert (meta["trial"] == vid["trial"]).all()
y, g = meta["y"], meta["subject"]

def summary(x):
    """Simple hand-crafted clip descriptor: per-channel mean, std, min, max."""
    x = x.reshape(len(x), x.shape[1], -1)
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)

feats = {
    "video (R2+1D pooled)": vid["pooled"],
    "IMU (summary stats)": summary(meta["X_imu"]),
    "skeleton (summary stats)": summary(meta["X_skel"]),
}
feats["all three concatenated"] = np.concatenate(list(feats.values()), axis=1)

for name, X in feats.items():
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    acc = cross_val_score(clf, X, y, groups=g, cv=LeaveOneGroupOut())
    print(f"{name:26s} dim={X.shape[1]:5d}  LOSO {acc.mean():.3f} +/- {acc.std():.3f}  "
          f"worst subj {acc.min():.3f}")
