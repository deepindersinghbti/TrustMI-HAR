"""Phase E, step 1: TMC and ECML on exactly the same clips and conditions as our certificate (no training).

Same folds, parts (val/test), seeds and conditions as cert_experts.py; the operators are deterministic,
so every input is identical. Clean log-probabilities are checked against each run's stored predictions.
One row per (method, seed, clip, condition); every signal is label-free:
  pred, msp        the method's combined answer and its probability
  u                combined uncertainty K / sum(alpha): the trust score of both papers
  conflict         mean pairwise conflictive degree between the sensors' opinions (ECML's measure)
  ev_u_<m>, ev_conf_<m>, ev_pred_<m>, ev_supp_<m>
                   sensor m's own uncertainty, confidence, answer, and probability for the combined answer
  ev_cdeg_<m>      mean conflictive degree of sensor m with the other sensors ("most in conflict")
  drop_pred_<m>    the combined answer with sensor m left out of the fusion (their natural "drop")
Labels (y, correct, donor_y) are stored only for scoring.
Output: results/certificate/evidential_<dataset>_<folds>_s<seeds>[_att-...].parquet
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cert_signals import CFG, run_dir, repro
from cert_score import auroc
from eval_operators import dataset_cls
from trustmi.models.evidential import EvidentialLateFusion, fuse_tmc, fuse_ecml
from trustmi.operators import OperatorBank
from cert_conditions import (build_fault_conditions, build_nowatch_conditions, build_realdrop_conditions,
                             conditions_suffix, parts_for, dataset_for, folds_for)
from trustmi.device import DEVICE, autocast

# run name -> (method, sensors in the order train.py used; ECML's fusion order depends on it)
METHODS = dict(d1=dict(tmc=("tmc", ["video", "skel", "imu"]), ecml=("ecml", ["video", "skel", "imu"])),
               d2=dict(tmc_s=("tmc", ["watch", "phone"]), ecml_s=("ecml", ["watch", "phone"])),
               d1v2=dict(tmc=("tmc", ["video", "skel", "imu"]), ecml=("ecml", ["video", "skel", "imu"])),   # journal
               d2v=dict(tmc_s3=("tmc", ["watch", "phone", "video"]), ecml_s3=("ecml", ["watch", "phone", "video"])),
               d3=dict(tmc_w=("tmc", ["arms", "trunk", "legs"]), ecml_w=("ecml", ["arms", "trunk", "legs"])))
PAPER = dict(tmc="tmc_p", ecml="ecml_p", tmc_s="tmc_sp", ecml_s="ecml_sp")   # --paper-settings run names


def load_ev(run, method, mods, K):
    cfg = json.loads((run / "config.json").read_text())
    model = EvidentialLateFusion(mods, method, n_classes=K, d=cfg.get("d", 128), layers=cfg.get("layers", 2))
    ck = torch.load(run / "ckpt_best.pt", map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)                    # strict: any mismatch stops here
    return model.to(DEVICE).eval()


@torch.no_grad()
def evidences(model, x, lens=None, bs=256):
    """Each sensor's evidence, in the same precision as train.py's evaluation (bf16 autocast)."""
    n = len(next(iter(x.values())))
    out = [[] for _ in model.mods]
    for a in range(0, n, bs):
        b = {m: v[a:a + bs].to(DEVICE) for m, v in x.items()}
        if lens is not None:
            b["len"] = lens[a:a + bs].to(DEVICE)
        with autocast():
            ev = model.evidences(b)
        for i, e in enumerate(ev):
            out[i].append(e.float().cpu())
    return [torch.cat(o).double() for o in out]


def fuse(method, ev):
    return fuse_tmc([e + 1 for e in ev]) if method == "tmc" else fuse_ecml(ev)


def signals(method, ev, mods, K):
    a_f = fuse(method, ev).numpy()
    p = a_f / a_f.sum(1, keepdims=True)
    pred, ar, V = p.argmax(1), np.arange(len(p)), len(mods)
    A = [(e + 1).numpy() for e in ev]
    P = [a / a.sum(1, keepdims=True) for a in A]
    U = [K / a.sum(1) for a in A]
    cd = lambda i, j: 0.5 * np.abs(P[i] - P[j]).sum(1) * (1 - U[i]) * (1 - U[j])   # conflictive degree
    cols = dict(pred=pred, msp=p.max(1), u=K / a_f.sum(1),
                conflict=np.mean([cd(i, j) for i, j in itertools.combinations(range(V), 2)], axis=0))
    for i, m in enumerate(mods):
        cols[f"ev_u_{m}"] = U[i]
        cols[f"ev_conf_{m}"] = P[i].max(1)
        cols[f"ev_pred_{m}"] = P[i].argmax(1)
        cols[f"ev_supp_{m}"] = P[i][ar, pred]
        cols[f"ev_cdeg_{m}"] = np.mean([cd(i, j) for j in range(V) if j != i], axis=0)
        cols[f"drop_pred_{m}"] = fuse(method, [ev[j] for j in range(V) if j != i]).numpy().argmax(1)
    return cols, np.log(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=CFG, required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--attack", default=None, help="comma list of sensors to swap (as in cert_experts.py)")
    ap.add_argument("--conditions", choices=("lies", "faults", "nowatch", "realdrop"), default="lies",
                    help="lies (default, unchanged), faults (docs/plan_faults_2026-11-11.md), "
                         "nowatch (d2v only, docs/plan_nowatch_2026-11-11.md) or "
                         "realdrop (d3 only, docs/plan_weargait_2026-11-11.md)")
    ap.add_argument("--no-video-faults", action="store_true",
                    help="faults without the video faults (smoke run before the video fault features exist)")
    ap.add_argument("--paper-settings", action="store_true", help="read the _p / _sp runs (papers' optimiser settings)")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    C, dname = CFG[args.dataset], args.dataset
    K, M = C["n_classes"], METHODS[args.dataset]
    if args.paper_settings:
        M = {PAPER[k]: v for k, v in M.items()}
    folds = list(range(C["n_folds"])) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    folds = folds_for(args.conditions, dname, folds)          # nowatch: only folds with no-watch clips
    seeds = [int(s) for s in args.seeds.split(",")]
    DS = dataset_cls(dname)
    out_dir = ROOT / "results/certificate"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.folds == "all" else "f" + "-".join(map(str, folds))
    attack = args.attack.split(",") if args.attack else C.get("attack", C["sensors"])
    suffix = (f"_att-{'-'.join(attack)}" if args.attack else "") + ("_paper" if args.paper_settings else "")
    suffix += conditions_suffix(args.conditions, not args.no_video_faults)
    out = out_dir / f"evidential_{dname}_{tag}_s{'-'.join(map(str, seeds))}{suffix}.parquet"
    rows, t0 = [], time.time()

    for fold in folds:
        tr = DS(fold, "train")
        models = {(name, s): load_ev(run_dir(dname, name, fold, s), meth, mods, K)
                  for name, (meth, mods) in M.items() for s in seeds}
        worst = 0.0
        for part in parts_for(args.conditions):
            ds = dataset_for(args.conditions, DS, dname, fold, part)
            base = {m: ds.x[m] for m in next(iter(M.values()))[1]}
            lens = ds.x["len"] if "len" in ds.x else None
            y = np.asarray(ds.y).astype(int); N = len(y)
            if args.conditions == "faults":              # the shared fault bench (cert_conditions.py)
                conds, gidx = build_fault_conditions(dname, ds, tr, fold, not args.no_video_faults)
                subj = np.asarray(ds.subject)
            elif args.conditions == "nowatch":           # the no-watch bench (cert_conditions.py)
                conds, gidx = build_nowatch_conditions(dname, ds, tr, fold)
                subj = np.asarray(ds.subject)
            elif args.conditions == "realdrop":          # the WearGait-PD real-dropout bench (cert_conditions.py)
                conds, gidx = build_realdrop_conditions(dname, ds, tr, fold)
                subj = np.asarray(ds.subject)
            else:                                        # the lie bench, exactly as before
                banks = {sen: OperatorBank(dname, sen, ds, tr, fold) for sen in attack}
                gidx, subj = banks[attack[0]].gidx, np.asarray(ds.subject)
                conds = {"clean": ({}, np.full(N, -1), np.ones(N, bool), "")}   # same construction as cert_experts.py
                for sen in attack:
                    for op in ("s_swap", "c_swap_same"):
                        xn, _, avail, info = banks[sen].apply(op)
                        conds[f"{op}:{sen}"] = ({sen: xn}, np.array([i["donor_y"] for i in info]), avail, sen)
                    if sen in C["sensors"]:                  # harmless changes need a physical model
                        xn, _, avail, _ = banks[sen].apply("h_all", level=2)
                        conds[f"h_all:{sen}"] = ({sen: xn}, np.full(N, -1), avail, sen)
            for cname, (repl, donor_y, avail, attacked) in conds.items():
                inp = dict(base); inp.update({k: torch.from_numpy(v) for k, v in repl.items()})
                for (name, s), mdl in models.items():
                    meth, mods = M[name]
                    cols, logp = signals(meth, evidences(mdl, inp, lens), mods, K)
                    if cname == "clean":
                        ok, msg = repro(run_dir(dname, name, fold, s), part, logp, y)
                        if not ok:
                            print(f"fold {fold} {part} {name} seed {s}: {msg}")
                            raise SystemExit("clean outputs do not match the stored run: stop and paste this")
                        worst = max(worst, float(msg.split()[-1]))
                    rows.append(pd.DataFrame(dict(
                        fold=fold, part=part, seed=s, method=name, clip=gidx, condition=cname, y=y,
                        subject=subj, attacked=attacked, donor_y=donor_y, available=avail,
                        correct=cols["pred"] == y, **cols)))
        print(f"fold {fold:2d}: stored runs reproduced (worst |log-prob diff| {worst:.4f}) | {time.time() - t0:.0f} s")

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out, index=False)
    t = df[(df.part == "test") & df.available].copy()
    t["group"] = np.where(t.condition == "clean", "normal",
                          np.where(t.condition.str.startswith("s_swap"), "lie",
                                   np.where(t.condition.str.startswith("f_"), "fault",
                                            np.where(t.condition.str.startswith(("nw_", "rd_")), "missing data",
                                                     "benign"))))
    pd.set_option("display.width", 160)
    print(f"\nSANITY ({dname}, test part) -> {out.relative_to(ROOT)}")
    print(t.groupby(["method", "group"])[["correct", "u", "conflict"]].mean().round(3).to_string())
    for name, g in t.groupby("method"):
        mods = np.array(M[name][1])
        lie = g.condition.str.startswith(("s_swap", "f_", "nw_", "rd_")).to_numpy()   # lies, faults, or missing data
        a = {k: round(auroc(v.to_numpy(), lie), 3) for k, v in (("1-msp", 1 - g.msp), ("u", g.u), ("conflict", g.conflict))}
        L = g[lie]
        nm = {r: round(float((mods[L[[f"{c}_{m}" for m in mods]].to_numpy().argmax(1)] == L.attacked.to_numpy()).mean()), 3)
              for r, c in (("most uncertain", "ev_u"), ("most in conflict", "ev_cdeg"))}
        print(f"{name}: rough pooled lie AUROC {a} | naming {nm}")


if __name__ == "__main__":
    main()
