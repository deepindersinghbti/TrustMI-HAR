"""Shared figure style for the TrustMI-HAR paper: serif text sized for an IEEE column, colour-blind-safe colours."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMN_IN = 3.5          # IEEE two-column layout: one column is 3.5 in wide


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,      # embed TrueType fonts (IEEE PDF eXpress)
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


PNG_DPI = 600            # 600 dpi: print quality; a full-column figure is about 2100 px wide


def save(fig, out, dpi=PNG_DPI):
    """Save the figure twice: vector PDF (for LaTeX) and a high-resolution PNG next to it."""
    out = out.with_suffix(".pdf")
    fig.savefig(out)
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    print("saved", out)
    print("saved", png, f"({dpi} dpi)")
