"""A3: run every counterfactual operator through trained models. One row per (seed, clip, target, op, level).

Before evaluating each fold it checks, for every seed:
  1. clean inputs reproduce the predictions stored in the run's preds_test.parquet
  2. r_mask truly hides the target modality: logits are unchanged when its content is replaced by noise
Results: results/operators/A3_<dataset>_<model>_<folds>.parquet
"""
import argparse, json, sys, time
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trustmi.models.fusion import FusionTransformer
from trustmi.operators import OperatorBank, OPS, LEVELS, FAMILY
from trustmi.device import DEVICE, autocast

MODELS = dict(b4=["video", "skel", "imu"], b4s=["watch", "phone"])
DATA = dict(d1=dict(n_classes=27, n_folds=8, targets=["imu"]),
            d2=dict(n_classes=35, n_folds=20, targets=["watch", "phone"]))


def dataset_cls(name):
    """Loader class for a dataset name; d1v2 / d2v are the journal versions (scripts/train.py DATASETS)."""
    if name in ("d1", "d1v2"):
        from trustmi.data.utd import UTDMHAD
        return UTDMHAD if name == "d1" else partial(UTDMHAD, video="v2")
    if name in ("d2", "d2v"):
        from trustmi.data.mmact import MMAct
        return MMAct if name == "d2" else partial(MMAct, video=True)
    raise ValueError(f"unknown dataset {name!r}")


def load_model(run, mods, n_classes):
    cfg = json.loads((run / "config.json").read_text())
    model = FusionTransformer(mods, n_classes=n_classes, d=cfg.get("d", 128), layers=cfg.get("layers", 2))
    ck = torch.load(run / "ckpt_best.pt", map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)                    # strict: any architecture mismatch stops here
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, x, lens=None, absent=None, bs=256):
    n = len(next(iter(x.values())))
    out = []
    for a in range(0, n, bs):
        b = {m: v[a:a + bs].to(DEVICE) for m, v in x.items()}
        if lens is not None:
            b["len"] = lens[a:a + bs].to(DEVICE)
        if absent is not None:
            b["absent"] = {m: v[a:a + bs].to(DEVICE) for m, v in absent.items()}
        with autocast():                                        # same precision as train.py's evaluation (bf16)
            out.append(model(b).float().cpu())
    return torch.cat(out)


LOGIT_TOL = 0.5   # bf16 rounding moves logits by ~0.01-0.1; a real mismatch (wrong checkpoint,
                  # normalisation or padding) moves them by whole units on many clips


def check_repro(run, logits, y):
    """Compare our clean logits with the logits train.py stored for the same clips (same order).
    If every logit is within LOGIT_TOL, any changed prediction is necessarily a near-tie:
    a flip needs the stored top-2 gap to be at most 2 x the largest logit difference."""
    old = pd.read_parquet(run / "preds_test.parquet")
    if not np.array_equal(old["true"].to_numpy(), y):
        return False, "stored rows are in a different clip order (true labels differ)"
    L = np.stack(old["logits"].to_numpy()).astype(np.float64)
    d = np.abs(logits - L)
    mism = np.flatnonzero(logits.argmax(1) != old["pred"].to_numpy())
    s = np.sort(L, 1); gap = s[:, -1] - s[:, -2]
    msg = (f"max |logit diff| {d.max():.4f} (median {np.median(d):.5f}) | "
           f"{len(y) - len(mism)}/{len(y)} preds identical"
           + (f", changed preds had stored top-2 gaps {np.round(gap[mism], 4).tolist()}"
              if len(mism) else ""))
    return bool(d.max() < LOGIT_TOL), msg


def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATA, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--folds", default="all", help="'all' or comma list, e.g. 0,1")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False                    # as in train.py
    D, mods = DATA[args.dataset], MODELS[args.model]
    folds = list(range(D["n_folds"])) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    DS = dataset_cls(args.dataset)
    out_dir = ROOT / "results/operators"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.folds == "all" else "f" + "-".join(map(str, folds))
    out = out_dir / f"A3_{args.dataset}_{args.model}_{tag}_s{'-'.join(map(str, seeds))}.parquet"
    rows, t0 = [], time.time()

    for fold in folds:
        tr, te = DS(fold, "train"), DS(fold, "test")
        base = {m: te.x[m] for m in mods}
        lens = te.x["len"] if "len" in te.x else None
        banks = {t: OperatorBank(args.dataset, t, te, tr, fold) for t in D["targets"]}
        gidx = banks[D["targets"][0]].gidx
        y = np.asarray(te.y).astype(int)
        models, clean = {}, {}
        for s in seeds:
            run = ROOT / "runs" / f"BASE_{args.dataset}_{args.model}_loso_f{fold:02d}_s{s}"
            models[s] = load_model(run, mods, D["n_classes"])
            logits = predict(models[s], base, lens).numpy().astype(np.float64)
            p = torch.softmax(torch.from_numpy(logits), 1).numpy()
            clean[s] = p
            ok, msg = check_repro(run, logits, y)
            print(f"fold {fold:2d} seed {s}: clean acc {(p.argmax(1) == y).mean():.3f} | {msg}")
            if not ok:
                raise SystemExit("clean logits do not match the stored run: stop and paste this")
            for t in D["targets"]:
                ab = {t: torch.ones(len(y), dtype=torch.bool)}
                noisy = dict(base); noisy[t] = torch.randn_like(base[t])
                d = (predict(models[s], base, lens, ab) - predict(models[s], noisy, lens, ab)).abs().max().item()
                if d > 1e-4:
                    raise SystemExit(f"r_mask leaks: hiding '{t}' still lets its content change logits by {d:.2e}")
        print(f"fold {fold:2d}: r_mask hides {D['targets']} completely (all seeds)")

        for t, bank in banks.items():
            for op in OPS:
                for lv in (sorted(LEVELS) if op.startswith("h_") else [0]):
                    xn, absent, avail, info = bank.apply(op, level=lv or 2)
                    if not avail.any():
                        continue
                    inp = dict(base); inp[t] = torch.from_numpy(xn)
                    ab = {t: torch.from_numpy(absent)} if absent.any() else None
                    donor_y = np.array([i.get("donor_y", -1) for i in info])
                    for s in seeds:
                        p = torch.softmax(predict(models[s], inp, lens, ab), 1).numpy()
                        pc = clean[s]
                        pred, pred_c = p.argmax(1), pc.argmax(1)
                        k = np.flatnonzero(avail)
                        rows.append(pd.DataFrame(dict(
                            dataset=args.dataset, model=args.model, fold=fold, seed=s,
                            subject=np.asarray(te.subject)[k], clip=gidx[k], y=y[k],
                            target=t, op=op, family=FAMILY[op], level=lv,
                            donor_y=donor_y[k], pred_clean=pred_c[k], pred=pred[k],
                            correct_clean=(pred_c == y)[k], correct=(pred == y)[k],
                            flipped=(pred != pred_c)[k], follows_donor=(pred == donor_y)[k],
                            conf_clean=pc.max(1)[k], conf=p.max(1)[k],
                            p_clean_pred=p[np.arange(len(p)), pred_c][k], p_true=p[np.arange(len(p)), y][k],
                            entropy=entropy(p)[k])))
        print(f"fold {fold:2d}: done ({time.time() - t0:.0f} s elapsed)")

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out, index=False)
    g = df.groupby(["target", "op", "level"], sort=False)
    summ = pd.DataFrame(dict(
        n=g.size(), acc=g.correct.mean(), flip=g.flipped.mean(),
        follows_donor=g.follows_donor.mean(), conf=g.conf.mean(),
        conf_drop=g.conf_clean.mean() - g.conf.mean())).round(3)
    summ.loc[summ.index.get_level_values("op") != "s_swap", "follows_donor"] = np.nan
    pd.set_option("display.width", 140)
    print(f"\nSUMMARY ({args.dataset} {args.model}, folds {tag}, seeds {seeds}) -> {out.relative_to(ROOT)}")
    print(summ.to_string())


if __name__ == "__main__":
    main()
