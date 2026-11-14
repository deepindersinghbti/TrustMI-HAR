"""Addition 1 (docs/plan_additions_2026-11-14.md, 1.2): the share of readings actually clipped by each saturation
level, for every sensor that gets saturation ("a level can clip almost nothing"; reported, never tuned).

  python scripts/faults2_clip_share.py          all folds of UTD-MHAD, MMAct, WearGait-PD (test part)
Run it when no queue is running: like every loader here, it rewrites the fold's normalisation file.
Writes results/certificate/faults2_clip_share.csv (one row per dataset, fold, sensor, level) and prints the
pooled shares. Deterministic (saturation has no random draw); no model is run.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from eval_operators import dataset_cls                       # the loaders every bench script uses
from cert_signals import CFG
from trustmi.faults2 import FaultBank2, APPLIES2, LEVELS, SAT

DATASETS = ("d1v2", "d2v", "d3")


def main():
    rows = []
    for d in DATASETS:
        DS = dataset_cls(d)
        sensors = [s for (dd, s), f in APPLIES2.items() if dd == d and "f_sat" in f and s != "video"]
        for fold in range(CFG[d]["n_folds"]):
            tr, te = DS(fold, "train"), DS(fold, "test")
            for s in sensors:
                bank = FaultBank2(d, s, te, tr, fold)
                for lv in LEVELS:
                    _, _, info = bank.apply_fault("f_sat", lv)
                    rows.append(dict(dataset=d, fold=fold, sensor=s, level=lv,
                                     clipped=sum(r["clipped"] for r in info), n_values=sum(r["n_values"] for r in info),
                                     clips_with_clipping=sum(r["clipped"] > 0 for r in info), n_clips=len(info)))
            print(f"{d} fold {fold}: done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results/certificate/faults2_clip_share.csv"
    df.to_csv(out, index=False)
    g = df.groupby(["dataset", "sensor", "level"])[["clipped", "n_values", "clips_with_clipping", "n_clips"]].sum()
    g["% readings clipped"] = 100 * g.clipped / g.n_values
    g["% clips with any clipping"] = 100 * g.clips_with_clipping / g.n_clips
    rng = {lv: f"+/-{SAT[lv]['acc_g']:g} g, +/-{SAT[lv]['gyro_dps']:g} deg/s" for lv in LEVELS}
    print("\nSATURATION: share of readings actually clipped (all folds pooled, test part)")
    print("  levels: " + "; ".join(f"L{lv} {r}" for lv, r in rng.items()))
    print(g[["% readings clipped", "% clips with any clipping"]].round(3).to_string())
    print(f"\nsaved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
