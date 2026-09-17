import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

meta = np.load("data/processed/utd_mhad_v1.npz", allow_pickle=True)
feat = np.load("data/features/video_r2plus1d18_v1.npz", allow_pickle=True)
assert (meta["trial"] == feat["trial"]).all(), "trial order mismatch"

X, y, g = feat["pooled"], meta["y"], meta["subject"]
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
acc = cross_val_score(clf, X, y, groups=g, cv=LeaveOneGroupOut())
print("per-subject accuracy:", acc.round(3))
print(f"LOSO mean {acc.mean():.3f} +/- {acc.std():.3f}   (chance = {1/27:.3f})")
