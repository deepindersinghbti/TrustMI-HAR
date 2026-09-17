import itertools
import pandas as pd

df = pd.read_csv("data/utd_mhad_lengths.csv")

print("Frame/sample counts per modality:")
print(df[["iner_n", "skel_n", "rgb_n"]].describe().round(1).to_string())

have = set(zip(df.action, df.subject, df.rep))
missing = [f"a{a}_s{s}_t{t}" for a, s, t in itertools.product(range(1, 28), range(1, 9), range(1, 5))
           if (a, s, t) not in have]
print("\nMissing trials:", missing)

df["skel_fps_vs_rgb"] = df.skel_n / df.rgb_sec
print("\nSkeleton rate implied by RGB duration:")
print(df.skel_fps_vs_rgb.describe().round(2).to_string())

df["dur_gap"] = df.iner_sec - df.rgb_sec
print("\nIMU minus RGB duration (s), by placement group:")
print(df.groupby("group").dur_gap.describe().round(2).to_string())
