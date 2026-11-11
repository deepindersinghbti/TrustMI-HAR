"""Journal ablation, step 1 of 2: put the SENSOR-DROPOUT models through the certificate's test bench. NO training.

The dropout models (b4_sd50 / b4_sd25 on d1v2, b4s3_sd50 / b4s3_sd25 on d2v; queue D) were so far only run
through the lie tests on the TEST person. The certificate's policy is fitted on the VALIDATION person, so this
script asks the dropout model about every clip and condition of both parts, exactly as cert_experts.py does for
the standard model (same folds, parts, seeds and conditions; the conditions come from cert_conditions.py, so
the inputs are identical). Per row, all label-free:
  sd_pred, sd_msp              the dropout model's answer and confidence on the input as given
  sd_hide_pred_<sensor>,       its answer and confidence with that sensor hidden (the model's own `absent`
  sd_hide_msp_<sensor>           mask, the mechanism it was trained with) = the "drop this sensor" answer
  sd_exp_support_<sensor>      how much each single-sensor witness believes the DROPOUT model's answer
  exp_pred_<sensor>            the witness's own answer (stored only so the scorer can check that the witnesses
                                 here are the same as in experts_<dataset>_*.parquet)
Plus y, subject, attacked, donor_y, available, so the file stands alone.
Check before use: the dropout model's clean logits must reproduce the logits stored by train.py (val and test).

  python scripts/cert_dropout.py --dataset d1v2 --rate sd50                      full run
  python scripts/cert_dropout.py --dataset d1v2 --rate sd50 --folds 0 --seeds 0  smoke run
Output: results/certificate/dropout_<dataset>_<rate>_<folds>_s<seeds>.parquet
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from eval_operators import load_model, dataset_cls, MODELS as FUSED
from cert_signals import CFG, run_dir, repro, fwd
from cert_conditions import conditions_for, conditions_suffix
from cert_experts import EXPERTS
from trustmi.sensor_dropout import RATES

JOURNAL = ("d1v2", "d2v")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=JOURNAL, required=True)
    ap.add_argument("--rate", choices=sorted(RATES), required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--conditions", choices=("lies", "faults"), default="lies",
                    help="lies (default, unchanged) or faults (docs/plan_faults_2026-11-11.md)")
    ap.add_argument("--no-video-faults", action="store_true",
                    help="faults without the video faults (smoke run before the video fault features exist)")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    C, dname = CFG[args.dataset], args.dataset
    model = f"{C['model']}_{args.rate}"                 # b4_sd50, b4s3_sd25, ...
    mods, E = FUSED[model], EXPERTS[dname]
    if mods != FUSED[C["model"]]:
        raise SystemExit(f"{model} and {C['model']} list their sensors differently: stop and paste this")
    folds = list(range(C["n_folds"])) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    DS = dataset_cls(dname)
    out_dir = ROOT / "results/certificate"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.folds == "all" else "f" + "-".join(map(str, folds))
    out = out_dir / (f"dropout_{dname}_{args.rate}_{tag}_s{'-'.join(map(str, seeds))}"
                     f"{conditions_suffix(args.conditions, not args.no_video_faults)}.parquet")
    attack = C.get("attack", C["sensors"])
    rows, t0 = [], time.time()

    for fold in folds:
        tr = DS(fold, "train")
        sd = {s: load_model(run_dir(dname, model, fold, s), mods, C["n_classes"]) for s in seeds}
        experts = {(m, s): load_model(run_dir(dname, b, fold, s), [m], C["n_classes"])
                   for m, b in E.items() for s in seeds}
        for part in ("val", "test"):
            ds = DS(fold, part)
            base = {m: ds.x[m] for m in mods}
            lens = ds.x["len"] if "len" in ds.x else None
            y = np.asarray(ds.y).astype(int); N = len(y); ar = np.arange(N)
            conds, gidx = conditions_for(args.conditions, dname, ds, tr, fold, attack, C["sensors"],
                                         not args.no_video_faults)               # shared with cert_experts.py
            subj = np.asarray(ds.subject)
            for cname, (repl, donor_y, avail, attacked) in conds.items():
                inp = dict(base); inp.update({k: torch.from_numpy(v) for k, v in repl.items()})
                for s in seeds:
                    z0, p0 = fwd(sd[s], inp, lens)
                    if cname == "clean":
                        ok, msg = repro(run_dir(dname, model, fold, s), part, z0, y)
                        print(f"fold {fold:2d} {part:4s} seed {s}: {model} clean acc {(p0.argmax(1) == y).mean():.3f} | {msg}")
                        if not ok:
                            raise SystemExit("clean logits do not match the stored run: stop and paste this")
                    pred = p0.argmax(1)
                    cols = dict(fold=fold, part=part, seed=s, clip=gidx, condition=cname, y=y, subject=subj,
                                attacked=attacked, donor_y=donor_y, available=avail,
                                sd_pred=pred, sd_msp=p0.max(1))
                    for m in mods:                                  # the same model with sensor m hidden
                        ph = fwd(sd[s], inp, lens, {m: torch.ones(N, dtype=torch.bool)})[1]
                        cols[f"sd_hide_pred_{m}"] = ph.argmax(1)
                        cols[f"sd_hide_msp_{m}"] = ph.max(1)
                    for m in E:                                     # witnesses judging the dropout model's answer
                        pm = fwd(experts[(m, s)], inp, lens)[1]
                        cols[f"sd_exp_support_{m}"] = pm[ar, pred]
                        cols[f"exp_pred_{m}"] = pm.argmax(1)
                    rows.append(pd.DataFrame(cols))
        print(f"fold {fold:2d}: done ({time.time() - t0:.0f} s elapsed)")

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out, index=False)
    t = df[(df.part == "test") & df.available]
    acc = lambda col: (t[col] == t.y).groupby(t.condition, sort=False).mean()
    summ = pd.DataFrame({"n": t.groupby("condition", sort=False).size(), "as given": acc("sd_pred"),
                         **{f"hide {m}": acc(f"sd_hide_pred_{m}") for m in mods}}).round(3)
    pd.set_option("display.width", 160)
    print(f"\nSANITY, test part: accuracy of {model} on each condition, as given and with one sensor hidden"
          f" -> {out.relative_to(ROOT)}")
    print(summ.to_string())


if __name__ == "__main__":
    main()
