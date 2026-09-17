import json
import numpy as np, pandas as pd, scipy.io as sio

ROOT = "data/extracted/UTD-MHAD"
OUT = "data/processed/utd_mhad_v1.npz"
T_IMU, T_SKEL = 100, 40
ROOT_JOINT, NECK_JOINT = 3, 1          # 0-based: hip_center, shoulder_center

def resample(x, T):
    """x: (n, ...) -> (T, ...) by linear interpolation over the clip's own span."""
    n = x.shape[0]
    flat = x.reshape(n, -1)
    old, new = np.linspace(0, 1, n), np.linspace(0, 1, T)
    out = np.stack([np.interp(new, old, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape((T,) + x.shape[1:])

meta = pd.read_csv("data/utd_mhad_lengths.csv").sort_values(["subject", "action", "rep"])
X_imu, X_skel = [], []
for tid in meta.trial:
    imu = sio.loadmat(f"{ROOT}/Inertial/Inertial/{tid}_inertial.mat")["d_iner"].astype(np.float64)
    skel = sio.loadmat(f"{ROOT}/Skeleton/Skeleton/{tid}_skeleton.mat")["d_skel"].astype(np.float64)
    skel = skel.transpose(2, 0, 1)                         # (frames, 20, 3)
    skel = skel - skel[:, ROOT_JOINT:ROOT_JOINT + 1, :]    # root-centre every frame
    spine = np.linalg.norm(skel[:, NECK_JOINT, :], axis=1).mean()
    assert spine > 1e-6, f"zero spine length in {tid}"
    skel = skel / spine                                     # scale-normalise
    X_imu.append(resample(imu, T_IMU))
    X_skel.append(resample(skel, T_SKEL))

X_imu = np.stack(X_imu).astype(np.float32)     # (N, 100, 6): ax ay az [g], gx gy gz [deg/s]
X_skel = np.stack(X_skel).astype(np.float32)   # (N, 40, 20, 3): hip-centred, spine-scaled
assert np.isfinite(X_imu).all() and np.isfinite(X_skel).all()

np.savez_compressed(
    OUT, X_imu=X_imu, X_skel=X_skel,
    y=meta.action.values - 1,                  # labels 0..26
    subject=meta.subject.values, rep=meta.rep.values,
    group=meta.group.values, trial=meta.trial.values)

with open("data/processed/utd_mhad_v1.json", "w") as f:
    json.dump(dict(version="v1", T_imu=T_IMU, T_skel=T_SKEL,
                   imu_rate_hz=50, skel_rate_hz_observed=19.5,
                   resampling="linear, over each modality's own span",
                   skel_norm="root=hip_center(4), scale=mean spine length",
                   zscore="NOT applied; fit per fold on training subjects",
                   missing_trials=["a8_s1_t4", "a23_s6_t4", "a27_s8_t4"],
                   n_trials=len(meta)), f, indent=2)

print("X_imu ", X_imu.shape, "| X_skel", X_skel.shape, "| N", len(meta))
print("labels per class (min/max):", np.bincount(meta.action.values - 1).min(),
      np.bincount(meta.action.values - 1).max())
print("IMU acc |g| mean:", np.linalg.norm(X_imu[..., :3], axis=-1).mean().round(3))
print("IMU gyro range:", X_imu[..., 3:].min().round(1), X_imu[..., 3:].max().round(1))
print("skel root joint max |x|:", np.abs(X_skel[:, :, 3, :]).max())
print("skel neck distance mean:", np.linalg.norm(X_skel[:, :, 1, :], axis=-1).mean().round(3))
