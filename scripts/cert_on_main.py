"""Phase E, step 3: our certificate on top of a trust-aware main model (TMC or ECML) instead of B4.

The experts judge the MAIN model's answers (cert_experts.py --main <run>); everything else is our
certificate exactly as in cert_policy.py: COMBINED(1-msp, expert JS, lowest expert support) fitted on
validation, odd-one-out naming (UTD-MHAD), use / drop / abstain in revisions 1 and 2.
Drop, fixed before any result (2026-09-25):
  primary    the main model's own fusion without the named sensor (drop_pred_<sensor>, cert_evidential.py)
  secondary  our option-A backup models (configs/backups.json), reported alongside, labelled
Compared with the main model alone (scored as in cert_evidential_score.py) and with our certificate on B4.
Writes results/certificate/e_<ds>_on_<main>_per_subject.csv
"""
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cert_policy as CP
from cert_evidential_score import per_subject, ev_feats, ev_drop, ci, OUR_RULES, EV_RULES, EVFILE, RES

KEYS = ["fold", "part", "seed", "clip", "condition"]
EXPFILE = {"d1": "experts_d1_all_s0-1-2_att-imu-skel-video", "d2": "experts_d2_all_s0-1-2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("d1", "d2"), required=True)
    ap.add_argument("--main", required=True, help="evidential run name: tmc / ecml (d1), tmc_s / ecml_s (d2)")
    a = ap.parse_args(); ds, mn = a.dataset, a.main
    ex = pd.read_parquet(RES / f"{EXPFILE[ds]}_main-{mn}.parquet")
    ex = ex.assign(pred=ex.pred_fused)
    ev = pd.read_parquet(RES / f"{EVFILE[ds]}.parquet")
    ev = ev[ev.method == mn]
    drops = [c for c in ev.columns if c.startswith("drop_pred_")]
    m = ex.merge(ev[KEYS + ["pred"] + drops].rename(columns={"pred": "pred_ev"}), on=KEYS,
                 how="inner", validate="one_to_one")
    if len(m) != len(ex):
        raise SystemExit(f"{len(ex) - len(m)} expert rows have no matching {mn} row: stop and paste this")
    agree = float((m.pred == m.pred_ev).mean())
    print(f"{mn}: both files give the same main answer in {agree:.4%} of {len(m)} rows")
    if agree < 0.999:
        raise SystemExit("the two runs did not see the same inputs: stop and paste this")
    c_main, c_bk = f"certificate on {mn}", f"certificate on {mn}, option-A drop"
    per = {}
    for rev in (1, 2):
        per[(c_main, rev)] = per_subject(m, ds, CP.features, OUR_RULES, ev_drop, rev)
        if ds == "d1":
            per[(c_bk, rev)] = per_subject(m, ds, CP.features, OUR_RULES, CP.backup_answer, rev)
        per[(f"{mn} alone", rev)] = per_subject(ev, ds, ev_feats, EV_RULES, ev_drop, rev)
        per[("certificate on B4", rev)] = per_subject(CP.load(ds), ds, CP.features, OUR_RULES, CP.backup_answer, rev)
    systems = [k for k in dict.fromkeys(k[0] for k in per)]
    for rev in (1, 2):
        print(f"\n{ds}: revision {rev}" + (" (post hoc, labelled)" if rev == 2 else ""))
        print(f"  {'system':34s} lie_AUROC naming% lie_acc% normal_acc% answer80%")
        for sname in systems:
            q = per[(sname, rev)]
            nm = f"{100 * q.naming.mean():7.1f}" if ds == "d1" else "    n/a"
            print(f"  {sname:34s} {q.lie_auroc.mean():9.3f} {nm} {100 * q.acc_policy_lie.mean():8.1f} "
                  f"{100 * q.acc_policy_normal.mean():11.1f} {100 * q.all_cov80_cert_policy.mean():9.1f}")
        o, b = per[(c_main, rev)], per[(f"{mn} alone", rev)]
        print(f"  {c_main} minus {mn} alone: lie_AUROC {ci(o.lie_auroc - b.lie_auroc, 1, 3)} | lie_acc "
              f"{ci(o.acc_policy_lie - b.acc_policy_lie, 100)} | normal {ci(o.acc_policy_normal - b.acc_policy_normal, 100)}"
              f" | answer80 {ci(o.all_cov80_cert_policy - b.all_cov80_cert_policy, 100)}")
    out = pd.concat([v.assign(system=k[0], revision=k[1]) for k, v in per.items()]).reset_index()
    out.to_csv(RES / f"e_{ds}_on_{mn}_per_subject.csv", index=False)
    print(f"\nsaved: results/certificate/e_{ds}_on_{mn}_per_subject.csv")


if __name__ == "__main__":
    main()
