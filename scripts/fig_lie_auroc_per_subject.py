"""Fig. 2: lie-detection AUROC per test subject for 1-MSP, ECML, TMC, the witness-only certificate and ours (+MI).

Reads (revision 1 rows; lie AUROC is the same in both revisions), written by cert_mi_score.py:
  results/certificate/mi_d1_per_subject.csv   UTD-MHAD
  results/certificate/mi_d2_per_subject.csv   MMAct
1-MSP is the column lie_auroc_msp of the certificate rows (B4's own confidence on the same clips).
Squares and the numbers under the axis labels are means over test subjects.
Writes paper/conference/figs/lie_auroc_per_subject.pdf and .png (600 dpi) and prints the means it plots.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fig_common import set_style, save, COLUMN_IN

CERTS = ["C3 (paper certificate)", "REV3 = C3 + MI (main)"]                         # step 5 names
SYSTEMS = {"d1": ["ECML alone", "TMC alone", *CERTS], "d2": ["ECML_S alone", "TMC_S alone", *CERTS]}
LABELS = ["1$-$MSP", "ECML", "TMC", "Witn.", "Ours"]
# one colour per method (validated colour-blind-safe set); 1-MSP is the neutral baseline
COLORS = ["#8a8984", "#eb6834", "#1baf7a", "#9ec5f4", "#2a78d6"]
TITLES = {"d1": "UTD-MHAD (8 subjects)", "d2": "MMAct (20 subjects)"}


def table(root, ds):
    d = pd.read_csv(root / f"results/certificate/mi_{ds}_per_subject.csv")   # written by cert_mi_score.py
    d = d[d.revision == 1]
    ours = d[d.system == CERTS[-1]].set_index("subject")
    cols = [ours.lie_auroc_msp]
    for s in SYSTEMS[ds]:
        cols.append(d[d.system == s].set_index("subject").lie_auroc)
    t = pd.concat(cols, axis=1, keys=LABELS).sort_index()
    if t.isna().any().any():
        raise SystemExit(f"{ds}: missing values per subject")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.root / "paper/conference/figs/lie_auroc_per_subject.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 2.0), sharey=True)
    x = np.arange(len(LABELS))
    for ax, ds in zip(axes, ("d1", "d2")):
        t = table(a.root, ds)
        for _, row in t.iterrows():                      # one light gray line per test subject
            ax.plot(x, row.values, color="0.80", lw=0.6, zorder=1)
            ax.scatter(x, row.values, s=5, c=COLORS, alpha=0.75, linewidths=0, zorder=2)
        m = t.mean()
        ax.plot(x, m.values, color="0.25", lw=1.2, zorder=3)
        ax.scatter(x, m.values, s=26, c=COLORS, marker="s", edgecolors="black", linewidths=0.6, zorder=4)
        ax.set_xticks(x, [f"{lab}\n{v:.3f}" for lab, v in zip(LABELS, m.values)])   # mean under each label
        ax.tick_params(axis="x", labelsize=5.8, pad=1.5)                           # five labels per panel
        ax.set_title(TITLES[ds], pad=3)
        ax.set_xlim(-0.4, len(LABELS) - 0.6)
        ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        print(ds, "means:", {k: round(v, 3) for k, v in m.items()}, "subjects:", len(t))
    axes[0].set_ylabel("Lie detection AUROC")
    axes[0].set_ylim(0.5, 1.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, out)


if __name__ == "__main__":
    main()
