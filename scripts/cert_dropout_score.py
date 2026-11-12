"""Journal ablation, step 2 of 2: score the SENSOR-DROPOUT models. Scored exactly like the main journal numbers.

  python scripts/cert_dropout_score.py --dataset d1v2
  python scripts/cert_dropout_score.py --dataset d2v
  ... --tag f0_s0     the smoke-run files (code check only; never overwrites the full-run CSV)

No existing scorer is edited. This script IMPORTS the per-test-subject loop behind every number so far
(cert_evidential_score.per_subject) and the main certificate REV3 (signals and naming rules from
cert_mi_score.SYSTEMS), and only changes WHO ANSWERS:

  A  main                REV3 on the standard fused model; "drop" hands over to the separate backup model
                         of configs/backups.json (the system reported so far)
  B  hide, no retraining REV3 on the standard fused model; "drop" = the SAME standard model with the named
                         sensor hidden (hide_pred_<sensor> of signals_<dataset>_*.parquet). Control: is any
                         extra model or retraining needed for the drop action?
  C  dropout as backup   REV3 on the standard fused model; "drop" = the dropout model with the named sensor
                         hidden. One backup model instead of one per sensor.
  D  dropout everywhere  REV3 on the DROPOUT model: it answers, the witnesses judge its answer, and "drop" =
                         the same model with the named sensor hidden. One fused model in total.
C and D are scored for both rates (sd50 = main, sd25 = sensitivity; rates fixed before any result).
The columns "alone" are the fused model with no certificate (standard for A-C, dropout for D): this answers
whether dropout training by itself makes the model robust to a lying sensor.

CHECK FIRST (full run only): row A is recomputed here and must equal the stored numbers of
mi_<dataset>_per_subject.csv to 1e-9, or the script stops.
Writes results/certificate/dropout_<dataset>_per_subject.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cert_policy as CP
import cert_mi_score as MS
from cert_evidential_score import per_subject, ci, RES

KEYS = MS.KEYS
RATES = ("sd50", "sd25")


def rates(ds):
    """WearGait-PD has only the 50% dropout model (docs/plan_weargait_2026-11-11.md); the others both rates."""
    return ("sd50",) if ds == "d3" else RATES
A = "A main (separate backups)"
B = "B hide, no retraining"
FULL = "all_s0-1-2"


def merge_cols(df, other, cols, what):
    """Add `cols` of `other` to df by clip and condition; every row must find exactly one partner."""
    n = len(df)
    df = df.merge(other[KEYS + cols], on=KEYS, how="inner", validate="one_to_one")
    if len(df) != n:
        raise SystemExit(f"{what}: {n - len(df)} rows have no partner: stop and paste this")
    return df


def hide_answer(prefix):
    """A drop action: the answer of one model with the named sensor hidden (columns <prefix>_<sensor>)."""
    def f(d, ds, named):
        out = np.empty(len(d), dtype=int)
        for s in CP.EXP[ds]:
            k = named == s
            out[k] = d[f"{prefix}_{s}"].to_numpy()[k]
        return out
    return f


def load_all(ds, tag):
    sens = CP.EXP[ds]
    df = MS.add_mi(CP.load(ds), ds)                    # standard model, witnesses, backups, MI: as the main scoring
    sg = pd.read_parquet(RES / f"signals_{ds}_{tag}.parquet")
    sg = sg.rename(columns={"pred": "sig_pred", **{f"hide_pred_{m}": f"std_hide_pred_{m}" for m in sens}})
    df = merge_cols(df, sg, ["sig_pred"] + [f"std_hide_pred_{m}" for m in sens], f"{ds} signals")
    same = (df.pred == df.sig_pred).mean()
    if same < 0.999:
        raise SystemExit(f"{ds}: signals and experts files disagree on the fused answer ({same:.4f}): stop and paste this")
    for r in rates(ds):
        f = RES / f"dropout_{ds}_{r}_{tag}.parquet"
        if not f.exists():
            raise SystemExit(f"missing {f.relative_to(ROOT)}: run scripts/cert_dropout.py --dataset {ds} --rate {r} first")
        sd = pd.read_parquet(f)
        own = [c for c in sd.columns if c.startswith("sd_")]
        chk = [f"exp_pred_{m}" for m in sens]
        sd = sd.rename(columns={c: f"{r}_{c[3:]}" for c in own} | {c: f"{r}_chk_{c}" for c in chk} | {"y": f"{r}_y"})
        new = [f"{r}_{c[3:]}" for c in own] + [f"{r}_chk_{c}" for c in chk] + [f"{r}_y"]
        df = merge_cols(df, sd, new, f"{ds} {r}")
        if not (df.y == df[f"{r}_y"]).all():
            raise SystemExit(f"{ds} {r}: labels differ from the experts file: stop and paste this")
        wit = min((df[c] == df[f"{r}_chk_{c}"]).mean() for c in chk)
        if wit < 0.999:
            raise SystemExit(f"{ds} {r}: witness answers differ from the experts file ({wit:.4f}): stop and paste this")
        print(f"check ({ds} {r}): same clips, labels and witnesses as the main scoring (witness agreement {wit:.4f})")
    return df


def as_main(df, ds, r):
    """The dropout model answers: its answer, confidence and the witnesses' support for ITS answer."""
    return df.assign(pred=df[f"{r}_pred"], msp=df[f"{r}_msp"],
                     **{f"exp_support_{m}": df[f"{r}_exp_support_{m}"] for m in CP.EXP[ds]})


def check_main(per, ds):
    stored = pd.read_csv(RES / f"mi_{ds}_per_subject.csv")
    stored = stored[stored.system == MS.MAIN]
    worst = 0.0
    for rev in (1, 2):
        a = per[(A, rev)]
        b = stored[stored.revision == rev].set_index("subject").loc[a.index]
        for c in MS.CHECK_COLS:
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                raise SystemExit(f"{ds}: NaN pattern of {c} differs from mi_{ds}_per_subject.csv: stop and paste this")
            worst = max(worst, float(np.nanmax(np.abs(x - y))) if (~np.isnan(x)).any() else 0.0)
    if worst > 1e-9:
        raise SystemExit(f"{ds}: row A differs from the stored main numbers by {worst:.2e}: stop and paste this")
    print(f"check ({ds}): row A reproduces the stored main certificate exactly (largest difference {worst:.1e})")


def table(per, ds):
    names = list(dict.fromkeys(k[0] for k in per))
    for rev in (1, 2):
        print(f"\n{ds}: sensor-dropout ablation, revision {rev}" + (" (post hoc, labelled)" if rev == 2 else "")
              + "   [accuracy in %, mean over test people]")
        print(f"  {'system':30s} {'lie_AUROC':>9s} {'naming%':>8s} | {'alone: normal':>13s} {'lie':>6s} | "
              f"{'policy: normal':>14s} {'lie':>6s} {'ans80':>6s}")
        for s in names:
            q = per[(s, rev)]
            print(f"  {s:30s} {q.lie_auroc.mean():9.3f} {100 * q.naming.mean():8.1f} | "
                  f"{100 * q.acc_main_normal.mean():13.1f} {100 * q.acc_main_lie.mean():6.1f} | "
                  f"{100 * q.acc_policy_normal.mean():14.1f} {100 * q.acc_policy_lie.mean():6.1f} "
                  f"{100 * q.all_cov80_cert_policy.mean():6.1f}")
        a = per[(A, rev)]
        print("  paired difference to row A (this row minus A), mean [95% interval] people better / all:")
        for s in names[1:]:
            q = per[(s, rev)].loc[a.index]
            print(f"  {s:30s} policy lie {ci(q.acc_policy_lie - a.acc_policy_lie, 100)} | policy normal "
                  f"{ci(q.acc_policy_normal - a.acc_policy_normal, 100)} | ans80 "
                  f"{ci(q.all_cov80_cert_policy - a.all_cov80_cert_policy, 100)}")
            if s.startswith("D"):
                print(f"  {'':30s} alone lie {ci(q.acc_main_lie - a.acc_main_lie, 100)} | alone normal "
                      f"{ci(q.acc_main_normal - a.acc_main_normal, 100)} | lie AUROC "
                      f"{ci(q.lie_auroc - a.lie_auroc, 1, 3)} | naming {ci(q.naming - a.naming, 100)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=CP.JOURNAL, required=True)
    ap.add_argument("--tag", default=FULL, help=f"{FULL} = the full run; f0_s0 = smoke-run files (code check only)")
    args = ap.parse_args(); ds = args.dataset
    CP.TAG = args.tag
    MS.journal_files(ds)
    if ds not in CP.BACKUPS:
        raise SystemExit(f"configs/backups.json has no entry for {ds}: run scripts/choose_backups_journal.py first")
    pd.set_option("display.width", 220)
    df = load_all(ds, args.tag)
    feats, rules = MS.signals(MS.SYSTEMS[MS.MAIN][0]), MS.SYSTEMS[MS.MAIN][1]

    systems = {A: (df, CP.backup_answer), B: (df, hide_answer("std_hide_pred"))}
    for r in rates(ds):
        systems[f"C dropout as backup ({r})"] = (df, hide_answer(f"{r}_hide_pred"))
    for r in rates(ds):
        systems[f"D dropout everywhere ({r})"] = (as_main(df, ds, r), hide_answer(f"{r}_hide_pred"))
    per = {(name, rev): per_subject(d, ds, feats, rules, drop, rev)
           for name, (d, drop) in systems.items() for rev in (1, 2)}
    if args.tag == FULL:
        check_main(per, ds)
    else:
        print(f"smoke run ({args.tag}): the check against the stored main numbers is skipped")
    table(per, ds)
    out = RES / f"dropout_{ds if args.tag == FULL else ds + '_' + args.tag}_per_subject.csv"
    pd.concat([v.assign(system=k[0], revision=k[1]) for k, v in per.items()]).reset_index().to_csv(out, index=False)
    print(f"\nsaved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
