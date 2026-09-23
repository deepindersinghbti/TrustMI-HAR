"""Linear probes on MMAct sensor data (sanity check, untuned C, NOT a reported result).
1) How informative is each sensor group about the action?  2) Does compass heading give away the scene?"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = np.load("data/processed/mmact_sensor_v1.npz")
X, start, length = d["X"], d["start"], d["length"]
keep = np.flatnonzero(d["status"] == "main")
y, g, scene = d["y"][keep], d["subject"][keep], d["scene"][keep]


def clip(i):
    """15 channels: watch acc 0-2, phone acc 3-5, gyro 6-8, sin(heading,pitch,roll) 9-11, cos(...) 12-14."""
    x = X[start[i]: start[i] + length[i]]
    ang = x[:, 9:12]
    return np.concatenate([x[:, :9], np.sin(ang), np.cos(ang)], axis=1)


F = np.stack([np.concatenate([c.mean(0), c.std(0), c.min(0), c.max(0)]) for c in map(clip, keep)])
cols = lambda ch: np.concatenate([np.asarray(ch) + 15 * k for k in range(4)])

WATCH = [0, 1, 2]
PHONE = [3, 4, 5, 6, 7, 8, 10, 11, 13, 14]   # phone acc + gyro + pitch/roll (no heading)
HEADING = [9, 12]
PITCH_ROLL = [10, 11, 13, 14]


def probe(ch, target, name):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    p = cross_val_predict(clf, F[:, cols(ch)], target, groups=g, cv=LeaveOneGroupOut(), n_jobs=-1)
    subs = np.unique(g)
    acc = np.array([(p[g == s] == target[g == s]).mean() for s in subs])
    f1 = np.mean([f1_score(target[g == s], p[g == s], labels=np.unique(target[g == s]), average="macro")
                  for s in subs])
    print(f"{name:32s} dim {len(cols(ch)):3d} | acc {acc.mean():.3f} +/- {acc.std():.3f}"
          f" | worst subj {acc.min():.3f} | macro-F1 {f1:.3f} | errors {(p != target).sum()}")


print(f"ACTION ({len(keep)} main clips, 35 classes, chance ~0.03)")
probe(WATCH, y, "watch")
probe(PHONE, y, "phone (no heading)")
probe(WATCH + PHONE, y, "watch + phone (no heading)")
probe(WATCH + PHONE + HEADING, y, "watch + phone + heading")

maj = np.bincount(scene).max() / len(scene)
print(f"\nSCENE (4 scenes, majority-class baseline {maj:.3f})")
probe(HEADING, scene, "heading only")
probe(PITCH_ROLL, scene, "pitch + roll only")
