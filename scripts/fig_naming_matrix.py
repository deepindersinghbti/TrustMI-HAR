"""Fig. 3: which sensor the certificate names, by the sensor that actually carries the lie (UTD-MHAD).

Reads results/certificate/experts_d1_all_s0-1-2_att-imu-skel-video.parquet (scripts/cert_experts.py).
Rows used: test part, available, conditions s_swap:<sensor>; all folds and seeds pooled.
Named sensor = odd one out, argmax over exp_div_<m> (the rule chosen on validation in all 8 folds,
see logs/cert_attrib_v3.log). Each row is normalised to 100%.
Writes paper/conference/figs/naming_matrix.pdf and .png (600 dpi) and prints the matrix (compare with cert_attrib_v3.log).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fig_common import set_style, save, COLUMN_IN

SENS = ["imu", "skel", "video"]                 # same order as cert_attrib.py (matters only for ties)
NAMES = ["Inertial", "Skeleton", "Video"]


def matrix(root):
    df = pd.read_parquet(root / "results/certificate/experts_d1_all_s0-1-2_att-imu-skel-video.parquet")
    d = df[(df.part == "test") & df.available & df.condition.str.startswith("s_swap")]
    named = np.array(SENS)[d[[f"exp_div_{m}" for m in SENS]].to_numpy().argmax(1)]
    ct = pd.crosstab(d.attacked, named).reindex(index=SENS, columns=SENS, fill_value=0)
    return 100 * ct.div(ct.sum(axis=1), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.root / "paper/conference/figs/naming_matrix.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    m = matrix(a.root)
    print(m.round(1).to_string())
    set_style()
    fig, ax = plt.subplots(figsize=(0.62 * COLUMN_IN, 1.75))
    ax.imshow(m.values, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 50 else "black", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3), NAMES)
    ax.set_yticks(range(3), NAMES)
    ax.set_xlabel("Sensor named by the certificate")
    ax.set_ylabel("Sensor carrying the lie")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    fig.tight_layout()
    save(fig, out)


if __name__ == "__main__":
    main()
