"""Plan step 4: per-clip pointwise MI between sensors, for every clip and test-bench condition (no training).

  python scripts/cert_mi.py --dataset d1 --version own --attack imu,skel,video
  python scripts/cert_mi.py --dataset d2 --version own
Same folds, parts (val/test), seeds and conditions as cert_experts.py (shared cert_conditions.build_conditions),
so every row matches a row of the experts file on (fold, part, seed, clip, condition).

Per clip, label-free, using the matcher trained by train_mi.py for that fold and seed:
  pmi_<a>_<b>     pointwise MI of sensors a and b (nats); high = they belong together
  mi_min, s4      lowest pairwise value and its negative (s4 = the certificate's new signal; high = suspicious)
  mi_mean         mean over pairs
  mi_share_<m>    mean pointwise MI of sensor m with the others; naming rule: the LOWEST share is named
Reference bank (fixed in advance): BANK_K training clips of the fold, drawn once with seed BANK_SEED + fold, the
same for every seed and condition; on MMAct each bank clip is time-stretched to the judged clip's length.
Labels (y, donor_y) are stored only for scoring.
Output: results/certificate/mi_<dataset>_<version>_<folds>_s<seeds>[_att-...].parquet
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from trustmi.mi import pairwise_pmi, mi_signals, resample_to
from trustmi.device import DEVICE, autocast
from trustmi.operators import VARIABLE_LENGTH

BANK_K, BANK_SEED = 64, 777


@torch.no_grad()
def embed_all(model, X, mods, lens=None, bs=256):
    """Unit vectors {sensor: (N, k)} for every clip of X (dict of tensors on the model's device)."""
    model.eval()
    n = len(X[mods[0]]); out = {m: [] for m in mods}
    for a in range(0, n, bs):
        b = {m: X[m][a:a + bs] for m in mods}
        if lens is not None:
            b["len"] = lens[a:a + bs]
        with autocast():
            z = model.embed(b)
        for m in mods:
            out[m].append(z[m].float())
    return {m: torch.cat(v) for m, v in out.items()}


class Bank:
    """The fixed reference clips, embedded once (UTD-MHAD) or once per needed length (MMAct, cached)."""
    def __init__(self, model, Xb, mods, lens_b=None):
        self.model, self.Xb, self.mods, self.lens_b, self.cache = model, Xb, mods, lens_b, {}
        if lens_b is None:
            self.shared = embed_all(model, Xb, mods)

    def at_length(self, L):
        if L not in self.cache:
            dst = torch.full_like(self.lens_b, L)
            # only padded time series are stretched; video tokens are fixed-size and stay as they are
            # (exactly as the lie donors in trustmi.operators and the candidates in train_mi.gather)
            Xs = {m: resample_to(self.Xb[m], self.lens_b, dst) if m in VARIABLE_LENGTH else self.Xb[m]
                  for m in self.mods}
            self.cache[L] = embed_all(self.model, Xs, self.mods, dst)
        return self.cache[L]

    def for_clips(self, lens):
        """{sensor: (K, k)} shared, or {sensor: (N, K, k)} with each clip's bank at that clip's length."""
        if self.lens_b is None:
            return self.shared
        per = [self.at_length(int(L)) for L in lens.tolist()]
        return {m: torch.stack([p[m] for p in per]) for m in self.mods}


def clip_mi(model, X, mods, bank, lens=None):
    """All MI columns for the clips in X: dict of numpy arrays."""
    z = embed_all(model, X, mods, lens)
    pmi = pairwise_pmi(z, bank.for_clips(lens) if lens is not None else bank.for_clips(None), model.scale().float())
    sig = mi_signals(pmi, mods)
    cols = {f"pmi_{a}_{b}": v.cpu().numpy() for (a, b), v in pmi.items()}
    cols.update({k: v.cpu().numpy() for k, v in sig.items()})
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["d1", "d2", "d1v2", "d2v"], required=True)
    ap.add_argument("--version", choices=["own", "frozen"], required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--attack", default=None, help="comma list of sensors to swap (as in cert_experts.py)")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    from cert_signals import CFG
    from cert_conditions import build_conditions
    from eval_operators import dataset_cls
    from train_mi import MODS, build_model, run_name
    from cert_score import auroc
    C, dname, mods = CFG[args.dataset], args.dataset, MODS[args.dataset]
    folds = list(range(C["n_folds"])) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    attack = args.attack.split(",") if args.attack else C.get("attack", C["sensors"])
    DS, dev = dataset_cls(dname), DEVICE
    tag = "all" if args.folds == "all" else "f" + "-".join(map(str, folds))
    suffix = f"_att-{'-'.join(attack)}" if args.attack else ""
    out = ROOT / "results/certificate" / f"mi_{dname}_{args.version}_{tag}_s{'-'.join(map(str, seeds))}{suffix}.parquet"
    rows, t0 = [], time.time()

    for fold in folds:
        tr = DS(fold, "train")
        bidx = torch.as_tensor(np.random.default_rng(BANK_SEED + fold).choice(len(tr.y), BANK_K, replace=False))
        Xb = {m: tr.x[m][bidx].to(dev) for m in mods}
        lens_b = tr.x["len"][bidx].to(dev) if "len" in tr.x else None
        models = {}
        for s in seeds:
            run = ROOT / "runs" / run_name(dname, args.version, fold, s)
            if not (ROOT / "results/raw" / f"{run.name}.parquet").exists():
                raise SystemExit(f"{run.name} is not complete: train it first (train_mi.py)")
            mdl = build_model(dname, args.version, fold, s, mods, dev)
            mdl.load_state_dict(torch.load(run / "ckpt_best.pt", map_location=dev))   # strict
            models[s] = mdl.eval()
        banks = {s: Bank(models[s], Xb, mods, lens_b) for s in seeds}
        for part in ("val", "test"):
            ds = DS(fold, part)
            base = {m: ds.x[m] for m in mods}
            lens = ds.x["len"].to(dev) if "len" in ds.x else None
            y = np.asarray(ds.y).astype(int); subj = np.asarray(ds.subject)
            conds, gidx = build_conditions(dname, ds, tr, fold, attack, C["sensors"])
            for cname, (repl, donor_y, avail, attacked) in conds.items():
                inp = dict(base); inp.update({k: torch.from_numpy(v) for k, v in repl.items()})
                inp = {m: inp[m].to(dev) for m in mods}
                for s in seeds:
                    cols = clip_mi(models[s], inp, mods, banks[s], lens)
                    rows.append(pd.DataFrame(dict(fold=fold, part=part, seed=s, clip=gidx, condition=cname, y=y,
                                                  subject=subj, attacked=attacked, donor_y=donor_y,
                                                  available=avail, **cols)))
        print(f"fold {fold:2d}: done ({time.time() - t0:.0f} s elapsed)", flush=True)

    df = pd.concat(rows, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    t = df[(df.part == "test") & df.available].copy()
    pd.set_option("display.width", 160)
    print(f"\nSANITY ({dname}, {args.version}, test part) -> {out.relative_to(ROOT)}")
    print("mean pointwise MI by condition (expected: clean ~ c_swap_same ~ h_all, s_swap clearly lower)")
    print(t.groupby("condition", sort=False)[[c for c in t.columns if c.startswith("pmi_")] + ["mi_min"]]
          .mean().round(3).to_string())
    lie = t.condition.str.startswith("s_swap").to_numpy()
    print(f"rough pooled lie AUROC of s4 (lies vs clean + controls): {auroc(t.s4.to_numpy(), lie):.3f}")
    if len(mods) >= 3:
        L = t[lie]
        named = np.array(mods)[L[[f"mi_share_{m}" for m in mods]].to_numpy().argmin(1)]
        print(f"rough naming accuracy (lowest MI share = the liar; chance {1 / len(mods):.3f}): "
              f"{(named == L.attacked.to_numpy()).mean():.3f}")
        print(pd.crosstab(L.attacked, named, normalize="index").round(3).to_string())


if __name__ == "__main__":
    main()
