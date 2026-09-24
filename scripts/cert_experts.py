"""Certificate v1b: ask the single-sensor EXPERT models about every clip and condition (no training).

Same folds, parts (val/test), seeds and conditions as cert_signals.py; the operators are deterministic,
so every input is identical (the scorer checks this by matching the fused answers row by row).
Per row, all label-free:
  exp_pred_<m>, exp_conf_<m>  the expert's own answer and confidence
  exp_support_<m>             how much the expert believes the FUSED model's answer (its prob for it)
  exp_js                      mean pairwise Jensen-Shannon divergence between the experts' distributions
  exp_disagree                True if the experts' answers are not all the same
Output: results/certificate/experts_<dataset>_<folds>_s<seeds>.parquet
"""
import argparse, itertools, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from eval_operators import load_model, dataset_cls, MODELS as FUSED
from cert_signals import CFG, run_dir, repro, fwd
from trustmi.operators import OperatorBank

EXPERTS = dict(d1=dict(imu="b1_imu", skel="b1_skel", video="b1_video"),
               d2=dict(watch="b1_watch", phone="b1_phone"))


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(np.clip(a, 1e-12, 1)) - np.log(np.clip(b, 1e-12, 1)))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=CFG, required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    C, dname = CFG[args.dataset], args.dataset
    mods, E = FUSED[C["model"]], EXPERTS[args.dataset]
    folds = list(range(C["n_folds"])) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    DS = dataset_cls(dname)
    out_dir = ROOT / "results/certificate"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.folds == "all" else "f" + "-".join(map(str, folds))
    out = out_dir / f"experts_{dname}_{tag}_s{'-'.join(map(str, seeds))}.parquet"
    rows, t0 = [], time.time()

    for fold in folds:
        tr = DS(fold, "train")
        fused = {s: load_model(run_dir(dname, C["model"], fold, s), mods, C["n_classes"]) for s in seeds}
        experts = {(m, s): load_model(run_dir(dname, b, fold, s), [m], C["n_classes"])
                   for m, b in E.items() for s in seeds}
        for part in ("val", "test"):
            ds = DS(fold, part)
            base = {m: ds.x[m] for m in mods}
            lens = ds.x["len"] if "len" in ds.x else None
            y = np.asarray(ds.y).astype(int); N = len(y); ar = np.arange(N)
            banks = {sen: OperatorBank(dname, sen, ds, tr, fold) for sen in C["sensors"]}
            gidx = banks[C["sensors"][0]].gidx
            conds = {"clean": {}}                        # identical construction to cert_signals.py
            for sen in C["sensors"]:
                for op in ("s_swap", "c_swap_same"):
                    conds[f"{op}:{sen}"] = {sen: banks[sen].apply(op)[0]}
                conds[f"h_all:{sen}"] = {sen: banks[sen].apply("h_all", level=2)[0]}
            for cname, repl in conds.items():
                inp = dict(base); inp.update({k: torch.from_numpy(v) for k, v in repl.items()})
                for s in seeds:
                    z0, p0 = fwd(fused[s], inp, lens)
                    if cname == "clean":
                        ok, msg = repro(run_dir(dname, C["model"], fold, s), part, z0, y)
                        print(f"fold {fold:2d} {part:4s} seed {s}: fused clean | {msg}")
                        if not ok:
                            raise SystemExit("clean logits do not match the stored run: stop and paste this")
                    pred = p0.argmax(1)
                    P = {m: fwd(experts[(m, s)], inp, lens)[1] for m in E}
                    cols = dict(fold=fold, part=part, seed=s, clip=gidx, condition=cname, pred_fused=pred)
                    for m, pm in P.items():
                        cols[f"exp_pred_{m}"] = pm.argmax(1)
                        cols[f"exp_conf_{m}"] = pm.max(1)
                        cols[f"exp_support_{m}"] = pm[ar, pred]
                    cols["exp_js"] = np.mean([js(P[a], P[b]) for a, b in itertools.combinations(P, 2)], axis=0)
                    ep = np.stack([P[m].argmax(1) for m in P])
                    cols["exp_disagree"] = (ep != ep[0]).any(0)
                    rows.append(pd.DataFrame(cols))
        print(f"fold {fold:2d}: done ({time.time() - t0:.0f} s elapsed)")

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out, index=False)
    t = df[df.part == "test"]
    print(f"\nSANITY ({dname}, test part) -> {out.relative_to(ROOT)}")
    print(t.groupby("condition", sort=False)[["exp_disagree", "exp_js"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
