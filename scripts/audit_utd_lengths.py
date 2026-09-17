import glob, os, re
import numpy as np, pandas as pd, scipy.io as sio, cv2

root = "data/extracted/UTD-MHAD"
pat = re.compile(r"a(\d+)_s(\d+)_t(\d+)_")
rows = []

for f in sorted(glob.glob(f"{root}/Inertial/Inertial/*_inertial.mat")):
    tid = os.path.basename(f).replace("_inertial.mat", "")
    a, s, t = map(int, pat.match(tid + "_").groups())

    iner = sio.loadmat(f)["d_iner"]

    skel_f = f"{root}/Skeleton/Skeleton/{tid}_skeleton.mat"
    skel = sio.loadmat(skel_f)["d_skel"] if os.path.exists(skel_f) else None

    rgb_f = f"{root}/RGB/RGB/{tid}_color.avi"
    cap = cv2.VideoCapture(rgb_f)
    rgb_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    rgb_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else np.nan
    cap.release()

    rows.append(dict(
        trial=tid, action=a, subject=s, rep=t,
        group="wrist" if a <= 21 else "thigh",
        iner_n=iner.shape[0],
        iner_nan=int(np.isnan(iner).sum()),
        skel_n=-1 if skel is None else skel.shape[2],
        skel_zero_frames=-1 if skel is None
            else int((np.abs(skel).sum(axis=(0, 1)) == 0).sum()),
        rgb_n=rgb_n, rgb_fps=rgb_fps,
    ))

df = pd.DataFrame(rows)
df["iner_sec"] = df.iner_n / 50.0                  # IMU is 50 Hz per the dataset page
df["rgb_sec"] = df.rgb_n / df.rgb_fps
df["implied_skel_fps"] = df.skel_n / df.iner_sec  # Kinect rate if spans are equal
df.to_csv("data/utd_mhad_lengths.csv", index=False)

print("trials:", len(df),
      "| subjects:", sorted(df.subject.unique()),
      "| actions:", df.action.nunique())
print("missing skeleton:", (df.skel_n < 0).sum(),
      "| unreadable RGB:", (df.rgb_n < 0).sum())
print("trials with NaN in inertial:", (df.iner_nan > 0).sum(),
      "| trials with all-zero skeleton frames:", (df.skel_zero_frames > 0).sum())
print()
print(df[["iner_sec", "rgb_sec", "implied_skel_fps", "rgb_fps"]].describe().round(2))
print("\nRGB fps values:", df.rgb_fps.value_counts().to_dict())
print("\nTrials where inertial and RGB durations differ by > 0.5 s:")
print(df.loc[(df.iner_sec - df.rgb_sec).abs() > 0.5,
             ["trial", "iner_sec", "rgb_sec", "skel_n"]].to_string(index=False))
print("\nTrials per subject:\n", df.groupby("subject").size().to_string())
print("\nTrials per placement group:\n", df.groupby("group").size().to_string())
