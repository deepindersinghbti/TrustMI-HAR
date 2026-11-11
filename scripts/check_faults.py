"""Fault plan, step 1 check on REAL data (no model): how strong is each fault on one fold's test person?

  python scripts/check_faults.py            fold 0 of both journal datasets (d1v2, d2v)
  python scripts/check_faults.py --fold 3

Per sensor, fault and level (test person only, nothing written to disk):
  change      RMS change of the valid samples in normalised units (1 = one training spread)
  clipped     share of drifted values that hit the training range (a real sensor would saturate there)
  over_half   timing faults only: share of clips whose shift is longer than half the clip
  drift_end   drift faults only: the full drift as a multiple of the training spread, per channel group
Also checks, on real data, the promises of tests/test_faults.py: padding and valid length unchanged.
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from eval_operators import dataset_cls
from trustmi.faults import FaultBank, faults_for, LEVELS, DRIFT, UNITS, G

SENSORS = {"d1v2": ["imu", "skel"], "d2v": ["watch", "phone"]}       # video faults come in plan step 4


def drift_ratio(bank, level):
    U, P, L, out = UNITS[(bank.base, bank.modality)], DRIFT[level], bank.layout, []
    if L.accel:
        size = P["acc_g"] * (1.0 if U["accel"] == "g" else G)
        out.append(f"accel {size / bank.sd[list(L.accel)].max():.2f}-{size / bank.sd[list(L.accel)].min():.2f}")
    if L.gyro:
        size = P["gyro_dps"] if U["gyro"] == "deg/s" else np.deg2rad(P["gyro_dps"])
        out.append(f"gyro {size / bank.sd[list(L.gyro)].max():.2f}-{size / bank.sd[list(L.gyro)].min():.2f}")
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    fold = ap.parse_args().fold
    for d, sensors in SENSORS.items():
        DS = dataset_cls(d)
        tr, te = DS(fold, "train"), DS(fold, "test")
        print(f"\n=== {d}, fold {fold}, test person {te.subject[0]}: {len(te.y)} clips")
        for s in sensors:
            b = FaultBank(d, s, te, tr, fold)
            valid = np.arange(b.T)[None, :] < b.lens[:, None]
            if b.angle_unit:
                print(f"  {s}: pitch / roll unit in the data = {b.angle_unit}")
            for fault in faults_for(d, s):
                for lv in LEVELS:
                    x, avail, info = b.apply_fault(fault, lv)
                    xx = x.reshape(b.N, b.T, b.C).astype(np.float64)
                    assert np.all(xx[~valid] == 0), f"{s} {fault} L{lv}: padding changed"
                    rms = float(np.sqrt(((xx - b.x)[valid] ** 2).mean()))
                    line = f"  {s:5s} {fault:7s} L{lv}: change {rms:5.2f}"
                    if fault == "f_drift":
                        clip = sum(i["clipped"] for i in info) / max(1, sum(i["n_values"] for i in info))
                        line += f" | clipped {100 * clip:5.2f}% | drift_end / spread: {drift_ratio(b, lv)}"
                    if fault == "f_gap":
                        line += f" | gap {np.mean([i['gap_len'] / b.lens[k] for k, i in enumerate(info)]):.3f} of clip"
                    if fault == "f_shift":
                        line += f" | over_half {100 * np.mean([i['shift_over_half'] for i in info]):4.1f}%"
                    print(line)
    print("\nOK: padding unchanged for every fault on real data")


if __name__ == "__main__":
    main()
