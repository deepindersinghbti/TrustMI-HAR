import numpy as np, pandas as pd

d = np.load("data/processed/utd_mhad_v1.npz", allow_pickle=True)
X, group, y = d["X_imu"], d["group"], d["y"]
gyro, acc = X[..., 3:], X[..., :3]

sat = np.abs(gyro) >= 999.0                       # at or near the ±1000 deg/s cap
clip_frac = sat.mean(axis=(1, 2))                 # fraction of saturated values per trial
print(f"trials with any gyro saturation: {(clip_frac > 0).sum()} / {len(X)}")
print(f"overall saturated gyro values: {sat.mean()*100:.2f}%")
print("\nSaturation by placement group (% of values):")
for g in ["wrist", "thigh"]:
    print(f"  {g}: {sat[group == g].mean()*100:.2f}%")

df = pd.DataFrame(dict(action=y + 1, frac=clip_frac))
print("\nTop 5 actions by mean saturated fraction (%):")
print((df.groupby("action").frac.mean() * 100).sort_values(ascending=False).head(5).round(2).to_string())

acc_first = np.linalg.norm(acc[:, :5, :], axis=-1).mean(axis=1)   # first 5 resampled steps
print(f"\n|acc| at clip start: median {np.median(acc_first):.3f} g, "
      f"IQR {np.percentile(acc_first,25):.3f}-{np.percentile(acc_first,75):.3f}")
print("accel saturation (|a| >= 7.9 g on any axis):", (np.abs(acc) >= 7.9).sum(), "values")
