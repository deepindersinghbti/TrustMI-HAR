"""Certificate v1, step 1: label-free warning signs for every clip, on the VALIDATION and TEST person of each fold.

CONDITION = what the model is shown:
  clean                 the unchanged input
  s_swap:<sensor>       that sensor replaced by the same person's DIFFERENT action (the lie to catch)
  c_swap_same:<sensor>  replaced by the same person's SAME action (benign control)
  h_all:<sensor>        a harmless level-2 change (benign control)
Signs, computed on that input WITHOUT labels:
  S1 stability  K harmless draws per sensor and level: how often the answer flips, and how much its prob drops
  S2 hide-one   each sensor hidden in turn (r_mask): flip, prob drop of the answer, total-variation shift
  baselines     msp, entropy, energy of the answer
  backups       answer of the option-A backup model for each droppable sensor (configs/backups.json)
Labels (y, donor_y, correct) are stored only for SCORING, never used by a sign.
Output: results/certificate/signals_<dataset>_<folds>.parquet
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from eval_operators import load_model, predict, dataset_cls, LOGIT_TOL, MODELS as FUSED
from trustmi.operators import OperatorBank, LEVELS

# sensors = the sensors with a physical model (they get harmless changes);
# attack  = the sensors lies are planted in by default (absent: the same as `sensors`, the conference setup)
CFG = dict(d1=dict(model="b4", n_classes=27, n_folds=8, sensors=["imu"]),
           d2=dict(model="b4s", n_classes=35, n_folds=20, sensors=["watch", "phone"]),
           # journal datasets: lies in EVERY sensor by default, so no --attack flag (and no _att- suffix) is needed
           d1v2=dict(model="b4", n_classes=27, n_folds=8, sensors=["imu"], attack=["imu", "skel", "video"]),
           d2v=dict(model="b4s3", n_classes=35, n_folds=20, sensors=["watch", "phone"],
                    attack=["watch", "phone", "video"]))
BACKUP_MODS = dict(b1_skel=["skel"], b1_phone=["phone"], b1_watch=["watch"])
K = 8               # harmless draws per sensor and level (fixed in advance, not tuned)
DRAW_SEED0 = 1000   # draws use bank seeds 1000..1000+K-1, independent of the h_all condition (seed 0)


def run_dir(ds, model, fold, seed):
    return ROOT / "runs" / f"BASE_{ds}_{model}_loso_f{fold:02d}_s{seed}"


def repro(run, part, logits, y):
    """Same logit-level check as eval_operators, for either stored part (val or test)."""
    old = pd.read_parquet(run / f"preds_{part}.parquet")
    if not np.array_equal(old["true"].to_numpy(), y):
        return False, "stored rows are in a different clip order"
    d = np.abs(logits - np.stack(old["logits"].to_numpy()).astype(np.float64))
    return bool(d.max() < LOGIT_TOL), f"max |logit diff| {d.max():.4f}"


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fwd(model, inp, lens, absent=None):
    z = predict(model, inp, lens, absent).numpy().astype(np.float64)
    return z, softmax(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=CFG, required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    C, dname = CFG[args.dataset], args.dataset
    mods = FUSED[C["model"]]
    folds = list(range(C["n_folds"])) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    # journal datasets have no entry here: their backup answers are stored by cert_experts.py (--extra-models)
    backups = {s: b for s, b in json.load(open(ROOT / "configs/backups.json")).get(dname, {}).items()}
    DS = dataset_cls(dname)
    out_dir = ROOT / "results/certificate"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.folds == "all" else "f" + "-".join(map(str, folds))
    out = out_dir / f"signals_{dname}_{tag}_s{'-'.join(map(str, seeds))}.parquet"
    rows, t0 = [], time.time()

    for fold in folds:
        tr = DS(fold, "train")
        fused = {s: load_model(run_dir(dname, C["model"], fold, s), mods, C["n_classes"]) for s in seeds}
        bk = {(sen, s): load_model(run_dir(dname, b, fold, s), BACKUP_MODS[b], C["n_classes"])
              for sen, b in backups.items() for s in seeds}
        for part in ("val", "test"):
            ds = DS(fold, part)
            base = {m: ds.x[m] for m in mods}
            lens = ds.x["len"] if "len" in ds.x else None
            y = np.asarray(ds.y).astype(int); N = len(y); ar = np.arange(N)
            banks = {sen: OperatorBank(dname, sen, ds, tr, fold) for sen in C["sensors"]}
            gidx, subj = banks[C["sensors"][0]].gidx, np.asarray(ds.subject)

            conds = {"clean": ({}, np.full(N, -1), np.ones(N, bool), "")}
            for sen in C["sensors"]:
                for op in ("s_swap", "c_swap_same"):
                    xn, _, avail, info = banks[sen].apply(op)
                    conds[f"{op}:{sen}"] = ({sen: xn}, np.array([i["donor_y"] for i in info]), avail, sen)
                xn, _, avail, _ = banks[sen].apply("h_all", level=2)
                conds[f"h_all:{sen}"] = ({sen: xn}, np.full(N, -1), avail, sen)
            for sen in C.get("attack", []):                  # journal: lies in the non-physical sensors too
                if sen not in C["sensors"]:
                    bank = OperatorBank(dname, sen, ds, tr, fold)
                    for op in ("s_swap", "c_swap_same"):
                        xn, _, avail, info = bank.apply(op)
                        conds[f"{op}:{sen}"] = ({sen: xn}, np.array([i["donor_y"] for i in info]), avail, sen)

            for cname, (repl, donor_y, avail, attacked) in conds.items():
                inp = dict(base); inp.update({k: torch.from_numpy(v) for k, v in repl.items()})
                cur = {sen: (repl[sen] if sen in repl else banks[sen].x.astype(np.float32)) for sen in C["sensors"]}
                res = {}
                for s in seeds:                                   # the answer on this input
                    z0, p0 = fwd(fused[s], inp, lens)
                    if cname == "clean":
                        ok, msg = repro(run_dir(dname, C["model"], fold, s), part, z0, y)
                        print(f"fold {fold:2d} {part:4s} seed {s}: clean acc {(p0.argmax(1) == y).mean():.3f} | {msg}")
                        if not ok:
                            raise SystemExit("clean logits do not match the stored run: stop and paste this")
                    pred0 = p0.argmax(1)
                    lse = np.log(np.exp(z0 - z0.max(1, keepdims=True)).sum(1)) + z0.max(1)
                    res[s] = dict(pred=pred0, p0=p0, cols=dict(
                        pred=pred0, correct=pred0 == y, msp=p0.max(1),
                        entropy=-(p0 * np.log(np.clip(p0, 1e-12, 1))).sum(1), energy=-lse))
                for sen in C["sensors"]:                          # S1: harmless draws on THIS input
                    b = copy.copy(banks[sen]); b.x = cur[sen].astype(np.float64)
                    for L in sorted(LEVELS):
                        flips = {s: np.zeros(N) for s in seeds}; drops = {s: np.zeros(N) for s in seeds}
                        for k in range(K):
                            b.seed = DRAW_SEED0 + k
                            xk = b.apply("h_all", level=L)[0]
                            ik = dict(inp); ik[sen] = torch.from_numpy(xk)
                            for s in seeds:
                                pk = fwd(fused[s], ik, lens)[1]
                                flips[s] += (pk.argmax(1) != res[s]["pred"]) / K
                                drops[s] += (res[s]["p0"][ar, res[s]["pred"]] - pk[ar, res[s]["pred"]]) / K
                        for s in seeds:
                            res[s]["cols"][f"stab_flip_{sen}_L{L}"] = flips[s]
                            res[s]["cols"][f"stab_drop_{sen}_L{L}"] = drops[s]
                for m in mods:                                    # S2: hide each sensor in turn
                    ab = {m: torch.ones(N, dtype=torch.bool)}
                    for s in seeds:
                        ph = fwd(fused[s], inp, lens, ab)[1]
                        pr, p0 = res[s]["pred"], res[s]["p0"]
                        res[s]["cols"][f"hide_flip_{m}"] = ph.argmax(1) != pr
                        res[s]["cols"][f"hide_drop_{m}"] = p0[ar, pr] - ph[ar, pr]
                        res[s]["cols"][f"hide_tv_{m}"] = 0.5 * np.abs(p0 - ph).sum(1)
                        res[s]["cols"][f"hide_pred_{m}"] = ph.argmax(1)       # answer without sensor m
                        res[s]["cols"][f"hide_msp_{m}"] = ph.max(1)           # and its confidence
                for sen in backups:                               # option-A backup answers
                    for s in seeds:
                        pb = fwd(bk[(sen, s)], inp, lens)[1]
                        res[s]["cols"][f"backup_pred_{sen}"] = pb.argmax(1)
                        res[s]["cols"][f"backup_conf_{sen}"] = pb.max(1)
                        res[s]["cols"][f"backup_correct_{sen}"] = pb.argmax(1) == y
                for s in seeds:
                    rows.append(pd.DataFrame(dict(
                        dataset=dname, fold=fold, part=part, seed=s, subject=subj, clip=gidx, y=y,
                        condition=cname, attacked=attacked, donor_y=donor_y, available=avail,
                        **res[s]["cols"])))
        print(f"fold {fold:2d}: done ({time.time() - t0:.0f} s elapsed)")

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out, index=False)
    t = df[(df.part == "test") & df.available]
    g = t.groupby("condition", sort=False)
    stab = [c for c in df.columns if c.startswith("stab_flip_") and c.endswith("_L2")]
    summ = pd.DataFrame(dict(n=g.size(), acc=g.correct.mean(), msp=g.msp.mean(),
                             stab_flip_L2=g[stab].mean().mean(axis=1)))
    hides = [f"hide_drop_{m}" for m in mods]
    suspect = np.array(mods)[t[hides].to_numpy().argmax(1)]
    hit = np.where(t.attacked.to_numpy() != "", (suspect == t.attacked.to_numpy()).astype(float), np.nan)
    summ["suspect_is_attacked"] = pd.Series(hit, index=t.index).groupby(t.condition, sort=False).mean()
    pd.set_option("display.width", 160)
    print(f"\nSANITY SUMMARY, test part ({dname}, folds {tag}, seeds {seeds}) -> {out.relative_to(ROOT)}")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
