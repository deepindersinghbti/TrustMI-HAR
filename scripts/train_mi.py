"""Plan step 3: train the MI matcher for one dataset / fold / seed / version.

  python scripts/train_mi.py --dataset d1 --version own --fold 0 --seed 0
  --version own     new sensor encoders (witness architecture), trained only for matching   [MAIN, fixed]
  --version frozen  the fold's witness encoders (runs/BASE_<d>_b1_<m>_...), frozen; only heads + scale train
  --smoke           quick check on 64 clips, 30 steps: loss must fall; nothing is saved

How a training example is built (docs/plan_mi_2026-11-04.md, section 3.1):
  * anchor = a training clip i; candidates = clip i itself + K other clips of the SAME subject and SAME group
    (UTD-MHAD: wrist/thigh placement; MMAct: scene), ANY action. No action labels are used anywhere.
  * every candidate is time-stretched to clip i's valid length exactly like the lie donors (MMAct; UTD-MHAD
    clips all have the same length), so length can never reveal a wrong partner.
  * the true partner is itself stretched to a random other length and back with probability 0.5 (MMAct),
    so that "this signal was stretched" cannot reveal a wrong partner either.
  * loss: InfoNCE over every ordered sensor pair; MI lower bound per pair: log(K+1) - CE (nats).
Training protocol = train.py (AdamW 3e-4, wd 0.05, cosine, bf16, grad clip 1.0, <= 60 epochs, patience 15),
except that early stopping uses the validation InfoNCE loss (there is no accuracy to stop on).
Validation candidates are drawn once with a fixed seed and no stretching of the true partner, so the
validation loss is comparable across epochs.
Output: runs/MI_<d>_<version>_loso_fXX_sY/{ckpt_best.pt, config.json, log.csv}
        results/raw/MI_<d>_<version>_loso_fXX_sY.parquet (written last; its presence means the run is complete)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from trustmi.mi import MIMatcher, sensor_encoder, info_nce_candidates, resample_to
from trustmi.operators import VARIABLE_LENGTH

MODS = dict(d1=["video", "skel", "imu"], d2=["watch", "phone"],          # same order as B4 / B4s
            d1v2=["video", "skel", "imu"], d2v=["watch", "phone", "video"],   # journal (D): as B4 / B4s3
            d3=["arms", "trunk", "legs"])                                       # journal: WearGait-PD, as b4w
WITNESS = dict(imu="b1_imu", skel="b1_skel", video="b1_video", watch="b1_watch", phone="b1_phone",
               arms="b1_arms", trunk="b1_trunk", legs="b1_legs")
N_CLASSES = dict(d1=27, d2=35, d1v2=27, d2v=35, d3=6)
POS_STRETCH_P = 0.5
VAL_SEED = 12345


def run_name(dname, version, fold, seed):
    return f"MI_{dname}_{version}_loso_f{fold:02d}_s{seed}"


class Candidates:
    """Groups of clips that may serve as wrong partners for each other: same subject AND same group."""
    def __init__(self, ds, K):
        self.K = K
        subj, grp = np.asarray(ds.subject), np.asarray(ds.group)
        keys = [f"{s}|{g}" for s, g in zip(subj, grp)]
        self.members = {}
        for i, k in enumerate(keys):
            self.members.setdefault(k, []).append(i)
        self.members = {k: np.asarray(v) for k, v in self.members.items()}
        self.key = keys
        self.lens = np.asarray(ds.x["len"]) if "len" in ds.x else None
        # WearGait-PD: clips of one labelled segment overlap in time (2 s clips, 1 s step), so they are never
        # each other's wrong partners. Datasets without `segment` (UTD-MHAD, MMAct) are unaffected.
        seg = getattr(ds, "segment", None)
        self.segment = None if seg is None else np.asarray(seg)

    def draw(self, idx, rng):
        """(B, K+1) clip indices (column 0 = the clip itself) and (B, K+1) validity mask."""
        C = np.zeros((len(idx), self.K + 1), np.int64); V = np.zeros_like(C, bool)
        for r, i in enumerate(idx):
            others = self.members[self.key[i]]
            others = others[others != i]
            if self.segment is not None:
                others = others[self.segment[others] != self.segment[i]]
            pick = rng.choice(others, size=min(self.K, len(others)), replace=False)
            C[r, 0], V[r, 0] = i, True
            C[r, 1:1 + len(pick)], V[r, 1:1 + len(pick)] = pick, True
            C[r, 1 + len(pick):] = i                           # unused slots: masked out of the loss
        return C, V

    def stretch_lengths(self, C, rng, stretch_pos):
        """Length each candidate is stretched FROM and TO, plus the optional detour length of the true partner."""
        if self.lens is None:
            return None
        src = self.lens[C]                                      # each candidate's own valid length
        dst = np.repeat(self.lens[C[:, :1]], C.shape[1], 1)     # the anchor's valid length
        detour = None
        if stretch_pos:
            flip = rng.random(len(C)) < POS_STRETCH_P
            other = self.lens[rng.integers(0, len(self.lens), len(C))]
            detour = np.where(flip, other, -1)                   # -1 = keep the true partner as it is
        return src, dst, detour


def gather(X, C, lens_info, mods):
    """Batch of all candidate clips, flattened to (B*(K+1), ...), each stretched to its anchor's length."""
    flat = torch.as_tensor(C.reshape(-1), device=X[mods[0]].device)
    batch = {m: X[m][flat] for m in mods}
    if lens_info is not None:
        src, dst, detour = (None if a is None else torch.as_tensor(a, device=flat.device) for a in lens_info)
        s, d = src.reshape(-1), dst.reshape(-1)
        for m in mods:
            if m not in VARIABLE_LENGTH:     # video tokens: fixed size (32 frames over the clip), never stretched
                continue
            x = batch[m]
            if detour is not None:                               # true partner: there and back again
                pos = torch.zeros_like(src, dtype=torch.bool); pos[:, 0] = detour >= 0
                pos = pos.reshape(-1)
                if pos.any():
                    mid = detour.repeat_interleave(src.shape[1])[pos]
                    x = x.clone()
                    x[pos] = resample_to(resample_to(x[pos], s[pos], mid), mid, d[pos])
            need = s != d
            if need.any():
                x = x.clone()
                x[need] = resample_to(x[need], s[need], d[need])
            batch[m] = x
        batch["len"] = d
    return batch


def loss_on(model, X, C, V, lens_info, mods):
    B, K1 = C.shape
    batch = gather(X, C, lens_info, mods)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X[mods[0]].is_cuda):
        z = model.embed(batch)
    zc = {m: v.float().reshape(B, K1, -1) for m, v in z.items()}
    return info_nce_candidates(zc, model.scale(), torch.as_tensor(V, device=zc[mods[0]].device))


def build_model(dname, version, fold, seed, mods, device):
    if version == "own":
        return MIMatcher({m: sensor_encoder(m) for m in mods}).to(device)
    from eval_operators import load_model                   # the fold's witnesses, same seed
    from cert_signals import run_dir
    enc = {}
    for m in mods:
        w = load_model(run_dir(dname, WITNESS[m], fold, seed), [m], N_CLASSES[dname])
        w.head = torch.nn.Identity()
        enc[m] = w
    return MIMatcher(enc, freeze_encoders=True).to(device)


@torch.no_grad()
def evaluate(model, X, Cv, Vv, lens_v, mods, bs=64):
    model.eval()
    tot, n, mi = 0.0, 0, {}
    for a in range(0, len(Cv), bs):
        sl = slice(a, a + bs)
        li = None if lens_v is None else tuple(None if x is None else x[sl] for x in lens_v)
        loss, info = loss_on(model, X, Cv[sl], Vv[sl], li, mods)
        w = len(Cv[sl]); tot += loss.item() * w; n += w
        for p, d in info.items():
            mi[p] = mi.get(p, 0.0) + d["mi_lb"].item() * w
    return tot / n, {f"{a}-{b}": v / n for (a, b), v in mi.items()}


def train(model, Xtr, ctr, Xva, cva, mods, args, device, log_every=None):
    """Returns (best_state, best_epoch, log rows). ctr / cva: Candidates of train / validation."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(args.seed)
    vr = np.random.default_rng(VAL_SEED + args.fold)
    Cv, Vv = cva.draw(np.arange(len(cva.key)), vr)
    lens_v = cva.stretch_lengths(Cv, vr, stretch_pos=False)
    best, best_state, best_ep, bad, log = np.inf, None, -1, 0, []
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(ctr.key))
        if getattr(args, "anchors_per_epoch", 0):           # a fresh random subset of clips every epoch
            order = order[:args.anchors_per_epoch]
        tot, n, te = 0.0, 0, time.time()
        for a in range(0, len(order), args.bs):
            idx = order[a:a + args.bs]
            C, V = ctr.draw(idx, rng)
            loss, _ = loss_on(model, Xtr, C, V, ctr.stretch_lengths(C, rng, stretch_pos=True), mods)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        sched.step()
        vl, vmi = evaluate(model, Xva, Cv, Vv, lens_v, mods)
        log.append(dict(epoch=ep, train_loss=tot / n, val_loss=vl, scale=model.scale().item(),
                        **{f"val_mi_{p}": v for p, v in vmi.items()}))
        if log_every and ep % log_every == 0:
            print(f"  epoch {ep:2d} train {tot / n:.3f} val {vl:.3f} " + " ".join(f"{p} {v:.2f}" for p, v in vmi.items())
                  + f" | {time.time() - te:.1f} s", flush=True)
        if vl < best:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    return best_state, best_ep, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=MODS, required=True)
    ap.add_argument("--version", choices=["own", "frozen"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=16, help="wrong partners per clip (fixed in advance)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--anchors-per-epoch", type=int, default=0, help="clips per epoch (0 = all training clips)")
    ap.add_argument("--log-every", type=int, default=0, help="print every N epochs (0 = only the final line)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.0,
                    help="cap this process's GPU memory (0 = no cap); lets MI share the GPU safely. "
                         "Changes only memory handling, never the results")
    args = ap.parse_args()
    if args.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    from train import set_seed, git_info
    from eval_operators import dataset_cls
    set_seed(args.seed)
    mods, DS, dev = MODS[args.dataset], dataset_cls(args.dataset), "cuda"
    tr, va = DS(args.fold, "train"), DS(args.fold, "val")
    Xtr = {k: v.to(dev) for k, v in tr.x.items()}; Xva = {k: v.to(dev) for k, v in va.x.items()}
    ctr, cva = Candidates(tr, args.k), Candidates(va, args.k)
    model = build_model(args.dataset, args.version, args.fold, args.seed, mods, dev)

    if args.smoke:                       # a small fixed subset; the training loss must fall
        sub = np.random.default_rng(0).choice(len(ctr.key), args.bs, replace=False)   # one real-size batch
        rng = np.random.default_rng(0)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3)
        C, V = ctr.draw(sub, rng); L = ctr.stretch_lengths(C, rng, stretch_pos=True)
        t0 = time.time()
        for step in range(30):
            model.train()
            loss, info = loss_on(model, Xtr, C, V, L, mods)
            opt.zero_grad(); loss.backward(); opt.step()
            if step in (0, 29):
                print(f"[smoke] step {step:2d} loss {loss.item():.3f} | " +
                      " ".join(f"{a}-{b} MI>={d['mi_lb'].item():.2f}" for (a, b), d in info.items()))
        print(f"[smoke] {args.dataset} {args.version}: 30 steps on {args.bs} clips x {args.k + 1} candidates in "
              f"{time.time() - t0:.1f} s; log(K+1) = {np.log(args.k + 1):.2f} nats is the ceiling")
        return

    name = run_name(args.dataset, args.version, args.fold, args.seed)
    run = ROOT / "runs" / name; run.mkdir(parents=True, exist_ok=True)
    sha, dirty = git_info()
    if dirty:
        print(f"WARNING: uncommitted changes; this run is not reproducible from {sha[:8]}")
    (run / "config.json").write_text(json.dumps(vars(args) | dict(
        exp_id=name, modalities=mods, pos_stretch_p=POS_STRETCH_P, val_seed=VAL_SEED, git_sha=sha,
        git_dirty=dirty, trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)), indent=2))
    t0 = time.time()
    state, ep, log = train(model, Xtr, ctr, Xva, cva, mods, args, dev, log_every=args.log_every)
    torch.save(state, run / "ckpt_best.pt")
    pd.DataFrame(log).to_csv(run / "log.csv", index=False)
    best = log[ep]
    rows = [dict(exp_id=name, dataset=args.dataset, version=args.version, fold=args.fold, seed=args.seed,
                 metric=k, value=v, best_epoch=ep, git_sha=sha)
            for k, v in best.items() if k.startswith("val_")]
    out = ROOT / "results/raw"; out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{name}.parquet.tmp"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, out / f"{name}.parquet")      # atomic: complete or absent
    mi = " ".join(f"{k[7:]} {v:.2f}" for k, v in best.items() if k.startswith("val_mi_"))
    print(f"{name}: best epoch {ep:2d} | val loss {best['val_loss']:.3f} | MI lower bound (nats) {mi} "
          f"| {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
