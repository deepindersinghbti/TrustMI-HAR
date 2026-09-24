"""Choose the option-A backup for each UTD-MHAD sensor by VALIDATION accuracy, with the rule fixed in advance:
the multi-sensor candidate (fused model without that sensor) must beat the single-sensor candidate by MORE
than 1 point of mean per-subject validation accuracy; otherwise the simpler single-sensor model is chosen.
The single-sensor candidate is the best remaining sensor on validation (IMU in every case).
Test accuracy is printed for information only and never used. Writes the d1 choices to configs/backups.json.
"""
import glob, json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDS = {"imu": ("b4_noimu", "b1_skel"), "skel": ("b4_noskel", "b1_imu"), "video": ("b4_novideo", "b1_imu")}
MARGIN = 0.01


def acc(model):
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/raw/BASE_d1_{model}_loso_f*_s*.parquet"))):
        d = pd.read_parquet(f)
        get = lambda part: d[(d.part == part) & (d.metric == "acc")].value.iloc[0]
        rows.append(dict(fold=int(re.search(r"_f(\d+)_s", f).group(1)), val=get("val"), test=get("test")))
    if len(rows) != 24:
        raise SystemExit(f"{model}: expected 24 runs, found {len(rows)}")
    return pd.DataFrame(rows).groupby("fold")[["val", "test"]].mean()      # per subject


def main():
    cfg_path = ROOT / "configs/backups.json"
    cfg = json.load(open(cfg_path))
    choice, evidence = {}, {}
    for sensor, (multi, single) in CANDS.items():
        a, b = acc(multi), acc(single)
        diff = a.val - b.val
        pick = multi if diff.mean() > MARGIN else single
        choice[sensor] = pick
        evidence[sensor] = {multi: round(100 * a.val.mean(), 1), single: round(100 * b.val.mean(), 1)}
        print(f"drop {sensor:5s}: {multi:10s} val {100 * a.val.mean():5.1f} test {100 * a.test.mean():5.1f} | "
              f"{single:8s} val {100 * b.val.mean():5.1f} test {100 * b.test.mean():5.1f} | "
              f"val diff {100 * diff.mean():+.1f} pts ({(diff > 0).sum()}/8 subjects) -> BACKUP {pick}")
    if choice["imu"] != cfg["d1"]["imu"]:
        raise SystemExit(f"IMU backup changed ({cfg['d1']['imu']} -> {choice['imu']}): stop and paste this")
    cfg["d1"] = choice
    cfg["evidence_val_acc_pct"]["d1"] = evidence
    cfg["_note"] = ("Certificate option A: the model that answers when a sensor is dropped. Chosen per sensor by "
                    "validation accuracy with a rule fixed in advance (multi-sensor candidate must beat the "
                    "single-sensor candidate by more than 1 point on validation, else the single-sensor model); "
                    "test accuracy never used. UTD-MHAD: scripts/choose_backups_d1.py.")
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"\nwritten {cfg_path.relative_to(ROOT)}: d1 = {choice}")


if __name__ == "__main__":
    main()
