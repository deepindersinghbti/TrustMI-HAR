"""Journal ablation: the certificate-v1 signs (harmless-change stability, hide-one-sensor) on the journal datasets.

  python scripts/cert_v1_score.py --dataset d1v2
  python scripts/cert_v1_score.py --dataset d2v
  python scripts/cert_v1_score.py --dataset d3      (added 13 Nov: sensor lists only)
  ... --tag f0_s0     the queue C smoke-run files (code check only; never overwrites the full-run CSVs)

The signs were computed by queue C (signals_<dataset>_*.parquet); only the scoring was missing. No existing
scorer is edited. The sign definitions are IMPORTED from cert_score.py (the conference scorer) and the
per-test-subject loop from cert_evidential_score.per_subject, so both tables are scored exactly like the
numbers reported so far. Everything is fitted on each fold's VALIDATION person and applied to its TEST person.

TABLE 1, each sign on its own (cert_score.score_task, as the conference v1/v2 tables):
  AUROC for catching lies and for catching the model's natural mistakes; paired difference to 1 - msp.
TABLE 2, as systems (lie AUROC, error AUROC, naming, use / drop / abstain; revisions 1 and 2):
  V1          1 - msp, S1 stability drop, S2 hide-one shift          naming: three hide-one rules
  C3          1 - msp, witness JS, lowest witness support            naming: odd one out
  REV3        C3 + MI  [MAIN]
  REV3 + V1   REV3 plus S1 and S2 (does the v1 test bench add anything once witnesses and MI are in?)
  All four use the same backups for the drop action (configs/backups.json).
S1 uses the sensors with a physical model only (UTD-MHAD: IMU; MMAct: watch, phone); skeleton and video have no
harmless change. Its strength level is chosen per fold on validation (best lie AUROC of S1), as in cert_score.py.

CHECKS: the signals and experts files must agree on the fused answer; C3's lie AUROC computed by the two
imported scorers must be equal to 1e-9; on the full run REV3 must equal mi_<dataset>_per_subject.csv to 1e-9.
Writes results/certificate/v1_<dataset>_auroc_per_subject.csv, v1_<dataset>_auroc_summary.csv,
       v1_<dataset>_per_subject.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cert_policy as CP
import cert_score as CS
import cert_mi_score as MS
from cert_evidential_score import per_subject, RES

FULL = "all_s0-1-2"
# the journal datasets, in cert_score's own tables (filled in here, cert_score.py itself is not edited)
PHYS = {"d1v2": ["imu"], "d2v": ["watch", "phone"],                      # sensors with a harmless change
        "d3": ["arms", "trunk", "legs"]}                                      # (d3: all IMUs, as cert_signals.py)
MODS = {"d1v2": ["video", "skel", "imu"], "d2v": ["watch", "phone", "video"], "d3": ["arms", "trunk", "legs"]}
V1, V1_ADD = "V1 (confidence + S1 + S2)", "REV3 + V1 signs"
V1_RULES = {"most support (hide_drop max)": ("hide_drop", "max"),
            "least support (hide_drop min)": ("hide_drop", "min"),
            "most confident without it (hide_msp max)": ("hide_msp", "max")}


def load(ds, tag):
    df = MS.add_mi(CP.load(ds), ds)                      # experts + MI: exactly the main scoring's rows
    sg = pd.read_parquet(RES / f"signals_{ds}_{tag}.parquet")
    cols = [c for c in sg.columns if c.startswith(("stab_", "hide_"))]
    n = len(df)
    df = df.merge(sg[MS.KEYS + ["pred"] + cols].rename(columns={"pred": "sig_pred"}), on=MS.KEYS, how="inner",
                  validate="one_to_one")
    if len(df) != n:
        raise SystemExit(f"{ds}: {n - len(df)} rows have no signals row: stop and paste this")
    agree = float((df.pred == df.sig_pred).mean())
    print(f"check ({ds}): signals merged; fused answers identical in {agree:.4%} of {n} rows")
    if agree < 0.999:
        raise SystemExit("the two runs did not see the same inputs: stop and paste this")
    return df


def add_v1_columns(df, ds):
    """v1_s1 = S1 stability drop at the level chosen on the fold's validation person; v1_s2 = S2 hide-one shift."""
    df = df.copy()
    df["v1_s2"] = df[[f"hide_tv_{m}" for m in MODS[ds]]].max(axis=1)
    df["v1_s1"] = np.nan
    chosen = {}
    for fold, g in df.groupby("fold"):
        va = g[(g.part == "val") & g.available]
        lie = va.condition.str.startswith("s_swap").to_numpy()
        s1 = lambda d, L: d[[f"stab_drop_{x}_L{L}" for x in PHYS[ds]]].mean(axis=1)
        auc = {L: CS.auroc(s1(va, L), lie, CS.MIN_CLASS) for L in CS.LEVELS}
        ok = {L: a for L, a in auc.items() if not np.isnan(a)}
        L = max(ok, key=ok.get) if ok else CS.DEFAULT_LEVEL
        chosen[fold] = L
        df.loc[g.index, "v1_s1"] = s1(g, L)
    return df, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=CP.JOURNAL, required=True)
    ap.add_argument("--tag", default=FULL, help=f"{FULL} = the full run; f0_s0 = smoke-run files (code check only)")
    args = ap.parse_args(); ds = args.dataset
    CP.TAG = args.tag
    MS.journal_files(ds)
    CS.SENS[ds], CS.MODS[ds], CS.EXPERT_MODS[ds] = PHYS[ds], MODS[ds], CP.EXP[ds]
    out_tag = ds if args.tag == FULL else f"{ds}_{args.tag}"
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    df = load(ds, args.tag)

    # ---- table 1: every sign on its own
    parts, lv = [], {}
    for task in ("LIE", "ERROR"):
        r, levels = CS.score_task(df, ds, task)
        parts.append(r); lv[task] = pd.Series(levels).value_counts().sort_index().to_dict()
    res = pd.concat(parts, ignore_index=True)
    summ = CS.summarise(res)
    print(f"\n{ds}: TABLE 1, each sign alone. AUROC per test subject, mean [95% CI]; minus_msp = paired difference "
          f"to the 1 - msp baseline")
    print(summ.drop(columns="dataset").to_string(index=False))
    print("harmless level chosen on validation (level: folds):", lv)

    # ---- table 2: as systems
    df, chosen = add_v1_columns(df, ds)
    sig, rules = MS.SYSTEMS[MS.MAIN]
    systems = {V1: (["s1", "v1_s1", "v1_s2"], V1_RULES),
               MS.REF: MS.SYSTEMS[MS.REF],
               MS.MAIN: (sig, rules),
               V1_ADD: (sig + ["v1_s1", "v1_s2"], rules)}
    per = {(name, rev): per_subject(df, ds, MS.signals(s), r, CP.backup_answer, rev)
           for name, (s, r) in systems.items() for rev in (1, 2)}

    a = res[(res.task == "LIE") & (res.score == "COMBINED v2 (experts+msp)")].set_index("subject").auroc
    b = per[(MS.REF, 1)].lie_auroc
    dev = float(np.nanmax(np.abs(a.loc[b.index] - b)))
    if dev > 1e-9:
        raise SystemExit(f"{ds}: the two scorers disagree on C3's lie AUROC by {dev:.2e}: stop and paste this")
    print(f"\ncheck ({ds}): C3's lie AUROC is the same in both scorers (largest difference {dev:.1e})")
    if args.tag == FULL:
        stored = pd.read_csv(RES / f"mi_{ds}_per_subject.csv")
        stored = stored[stored.system == MS.MAIN]
        worst = 0.0
        for rev in (1, 2):
            p = per[(MS.MAIN, rev)]
            q = stored[stored.revision == rev].set_index("subject").loc[p.index]
            for c in MS.CHECK_COLS:
                x, y = p[c].to_numpy(float), q[c].to_numpy(float)
                if not np.array_equal(np.isnan(x), np.isnan(y)):
                    raise SystemExit(f"{ds}: NaN pattern of {c} differs from mi_{ds}_per_subject.csv: stop and paste this")
                worst = max(worst, float(np.nanmax(np.abs(x - y))) if (~np.isnan(x)).any() else 0.0)
        if worst > 1e-9:
            raise SystemExit(f"{ds}: REV3 differs from the stored main numbers by {worst:.2e}: stop and paste this")
        print(f"check ({ds}): REV3 reproduces the stored main certificate exactly (largest difference {worst:.1e})")
    else:
        print(f"smoke run ({args.tag}): the check against the stored main numbers is skipped")
    print(f"S1 level used in table 2 (fold: level): {chosen}")
    MS.table(per, ds, [V1, V1_ADD], "TABLE 2, v1 signs as systems")

    res.to_csv(RES / f"v1_{out_tag}_auroc_per_subject.csv", index=False)
    summ.to_csv(RES / f"v1_{out_tag}_auroc_summary.csv", index=False)
    pd.concat([v.assign(system=k[0], revision=k[1]) for k, v in per.items()]).reset_index().to_csv(
        RES / f"v1_{out_tag}_per_subject.csv", index=False)
    print(f"\nsaved: results/certificate/v1_{out_tag}_auroc_per_subject.csv, v1_{out_tag}_auroc_summary.csv, "
          f"v1_{out_tag}_per_subject.csv")


if __name__ == "__main__":
    main()
