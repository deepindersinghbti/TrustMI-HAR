"""Journal datasets: choose the option-A backup (the model that answers when a sensor is dropped) for every
sensor of d1v2 (UTD-MHAD, whole-frame video) and d2v (MMAct with video), by VALIDATION accuracy only.

The rule is the one fixed on 2026-09-24 for the conference paper (scripts/choose_backups_d1.py), unchanged:
  * multi-sensor candidate  = the fused model trained without that sensor
  * single-sensor candidate = the best remaining single-sensor model on validation
  * the multi-sensor candidate is chosen only if it beats the single-sensor one by MORE than 1 point of
    mean per-subject validation accuracy; otherwise the simpler single-sensor model is chosen.
Test accuracy is printed for information only and never used.
Writes the d1v2 and d2v choices (and the validation evidence) into configs/backups.json; d1 and d2 stay as they are.
"""
import glob, json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDS = {   # dataset -> dropped sensor -> (multi-sensor candidate, single-sensor candidates)
    "d1v2": {"imu": ("b4_noimu", ["b1_skel", "b1_video"]), "skel": ("b4_noskel", ["b1_imu", "b1_video"]),
             "video": ("b4_novideo", ["b1_imu", "b1_skel"])},
    "d2v": {"watch": ("b4s_nowatch", ["b1_phone", "b1_video"]), "phone": ("b4s_nophone", ["b1_watch", "b1_video"]),
            "video": ("b4s", ["b1_watch", "b1_phone"])},
    # WearGait-PD (docs/plan_weargait_2026-11-11.md): same rule; one "subject" below = one fold of ~18 people
    "d3": {"arms": ("b4w_noarms", ["b1_trunk", "b1_legs"]), "trunk": ("b4w_notrunk", ["b1_arms", "b1_legs"]),
           "legs": ("b4w_nolegs", ["b1_arms", "b1_trunk"])},
}
N_RUNS = {"d1v2": 24, "d2v": 60, "d3": 30}          # folds x 3 seeds
MARGIN = 0.01


def acc(ds, model):
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/raw/BASE_{ds}_{model}_loso_f*_s*.parquet"))):
        d = pd.read_parquet(f)
        get = lambda part: d[(d.part == part) & (d.metric == "acc")].value.iloc[0]
        rows.append(dict(fold=int(re.search(r"_f(\d+)_s", f).group(1)), val=get("val"), test=get("test")))
    if len(rows) != N_RUNS[ds]:
        raise SystemExit(f"{ds} {model}: expected {N_RUNS[ds]} runs, found {len(rows)}: stop and paste this")
    return pd.DataFrame(rows).groupby("fold")[["val", "test"]].mean()      # per subject


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="d1v2,d2v",
                    help="comma-separated; the default reproduces the d1v2 / d2v choices exactly. WearGait: d3")
    todo = ap.parse_args().datasets.split(",")
    cfg_path = ROOT / "configs/backups.json"
    cfg = json.load(open(cfg_path))
    for ds, cands in CANDS.items():
        if ds not in todo:
            continue
        choice, evidence = {}, {}
        n = N_RUNS[ds] // 3
        print(f"\n{ds}  (validation accuracy, mean over {n} subjects; test shown for information only)")
        for sensor, (multi, singles) in cands.items():
            a = acc(ds, multi)
            s_acc = {m: acc(ds, m) for m in singles}
            single = max(singles, key=lambda m: s_acc[m].val.mean())       # best remaining sensor on validation
            b = s_acc[single]
            diff = a.val - b.val
            pick = multi if diff.mean() > MARGIN else single
            choice[sensor] = pick
            evidence[sensor] = {multi: round(100 * a.val.mean(), 1),
                                **{m: round(100 * s_acc[m].val.mean(), 1) for m in singles}}
            print(f"  drop {sensor:5s}: {multi:11s} val {100 * a.val.mean():5.1f} test {100 * a.test.mean():5.1f} | "
                  + " | ".join(f"{m:8s} val {100 * s_acc[m].val.mean():5.1f} test {100 * s_acc[m].test.mean():5.1f}"
                               for m in singles)
                  + f" | val diff to best single {100 * diff.mean():+.1f} pts ({(diff > 0).sum()}/{n} subjects)"
                    f" -> BACKUP {pick}")
        cfg[ds] = choice
        cfg["evidence_val_acc_pct"][ds] = evidence
    if "journal" not in cfg["_note"]:
        cfg["_note"] += (" Journal datasets d1v2 / d2v: scripts/choose_backups_journal.py, same rule; the "
                         "single-sensor candidate is the best remaining sensor on validation.")
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"\nwritten {cfg_path.relative_to(ROOT)}: " + ", ".join(f"{ds} = {cfg[ds]}" for ds in todo))


if __name__ == "__main__":
    main()
