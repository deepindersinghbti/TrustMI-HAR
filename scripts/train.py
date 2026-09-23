import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, platform, random, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trustmi.data.utd import UTDMHAD
from trustmi.data.mmact import MMAct
from trustmi.models.fusion import FusionTransformer

MODELS = dict(b1_video=["video"], b1_skel=["skel"], b1_imu=["imu"],
              b4=["video", "skel", "imu"],
              b1_watch=["watch"], b1_phone=["phone"], b4s=["watch", "phone"])
DATASETS = dict(d1=dict(cls=UTDMHAD, n_classes=27, data="utd_mhad_v1",
                        video_features="video_r2plus1d18_v1", split="utd_mhad_loso_v1"),
                d2=dict(cls=MMAct, n_classes=35, data="mmact_sensor_v1",
                        video_features=None, split="mmact_loso_v1"))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    except Exception:
        sha, dirty = "unknown", True
    return sha, dirty


def to_gpu(ds):
    return {k: v.cuda() for k, v in ds.x.items()}, ds.y.cuda()


def batches(X, y, bs, gen=None):
    order = torch.randperm(len(y), generator=gen) if gen is not None else torch.arange(len(y))
    for i in range(0, len(y), bs):
        j = order[i:i + bs].cuda()
        yield {k: v[j] for k, v in X.items()}, y[j]


@torch.no_grad()
def predict(model, X, y):
    model.eval()
    out = []
    for xb, _ in batches(X, y, 128):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(xb).float())
    return torch.cat(out)


def scores(logits, y):
    p = logits.softmax(-1)
    return dict(pred=logits.argmax(-1).cpu().numpy(),
                msp=p.max(-1).values.cpu().numpy(),
                entropy=(-(p * p.clamp_min(1e-12).log()).sum(-1)).cpu().numpy(),
                energy=torch.logsumexp(logits, -1).cpu().numpy(),
                loss=F.cross_entropy(logits, y).item())


def metrics(s, y):
    yt = y.cpu().numpy()
    return dict(acc=float((s["pred"] == yt).mean()),
                macro_f1=float(f1_score(yt, s["pred"], labels=np.unique(yt), average="macro")),
                nll=s["loss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--dataset", choices=DATASETS, default="d1")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="overfit-one-batch test (N6)")
    ap.add_argument("--shuffle_labels", action="store_true", help="negative control N1")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)          # non-deterministic backward
    torch.backends.cuda.enable_mem_efficient_sdp(False)  # non-deterministic backward
    torch.backends.cuda.enable_math_sdp(True)            # deterministic reference kernel
    set_seed(args.seed)
    mods = MODELS[args.model]

    D = DATASETS[args.dataset]
    tr, va, te = (D["cls"](args.fold, p) for p in ("train", "val", "test"))
    missing = [m for m in mods if m not in tr.x]
    if missing:
        raise SystemExit(f"model {args.model} needs {missing}, not available in {args.dataset}")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = to_gpu(tr), to_gpu(va), to_gpu(te)
    if args.shuffle_labels:
        g = torch.Generator().manual_seed(1000 + args.seed)
        ytr = ytr[torch.randperm(len(ytr), generator=g).cuda()]
    model = FusionTransformer(mods, n_classes=D["n_classes"], d=args.d, layers=args.layers).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.smoke:
        xb, yb = {k: v[:8] for k, v in Xtr.items()}, ytr[:8]
        model.train()
        for _ in range(300):
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[smoke] {args.model:9s} loss after 300 steps on 8 samples: {loss.item():.4f}")
        return

    block = "N1" if args.shuffle_labels else "BASE"
    exp_id = f"{block}_{args.dataset}_{args.model}_loso_f{args.fold:02d}_s{args.seed}"
    run = ROOT / "runs" / exp_id
    run.mkdir(parents=True, exist_ok=True)
    sha, dirty = git_info()
    if dirty:
        print(f"WARNING: uncommitted changes; this run is not reproducible from {sha[:8]}")
    cfg = vars(args) | dict(exp_id=exp_id, modalities=mods, config_tag="defaults_v0",
                            data=D["data"], video_features=D["video_features"],
                            split=D["split"], git_sha=sha, git_dirty=dirty)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    (run / "env.json").write_text(json.dumps(dict(
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), params=sum(p.numel() for p in model.parameters())), indent=2))

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    gen = torch.Generator().manual_seed(args.seed)
    best_f1, best_state, best_ep, bad, log, t0 = -1.0, None, -1, 0, [], time.time()

    for ep in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in batches(Xtr, ytr, args.bs, gen):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(yb); n += len(yb)
        sched.step()
        vm = metrics(scores(predict(model, Xva, yva), yva), yva)
        log.append(dict(epoch=ep, train_loss=total / n, **{f"val_{k}": v for k, v in vm.items()}))
        if vm["macro_f1"] > best_f1:
            best_f1, best_ep, bad = vm["macro_f1"], ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    torch.save(best_state, run / "ckpt_best.pt")
    pd.DataFrame(log).to_csv(run / "log.csv", index=False)

    rows = []
    for part, ds, X, y in [("val", va, Xva, yva), ("test", te, Xte, yte)]:
        lg = predict(model, X, y)
        s = scores(lg, y)
        pd.DataFrame(dict(
            exp_id=exp_id, trial=ds.trial, subject=ds.subject, group=ds.group,
            true=y.cpu().numpy(), pred=s["pred"], msp=s["msp"], entropy=s["entropy"],
            energy=s["energy"], logits=[r.tolist() for r in lg.cpu().numpy()],
        )).to_parquet(run / f"preds_{part}.parquet", index=False)
        for k, v in metrics(s, y).items():
            rows.append(dict(exp_id=exp_id, block=block, dataset=args.dataset, model=args.model,
                             fold=args.fold, seed=args.seed, part=part, condition="clean",
                             metric=k, value=v, best_epoch=best_ep, git_sha=sha,
                             config_tag="defaults_v0"))
    out = ROOT / "results/raw"
    out.mkdir(parents=True, exist_ok=True)
    os.sync()                                   # checkpoint, log and predictions reach the disk first
    tmp = out / f"{exp_id}.parquet.tmp"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, out / f"{exp_id}.parquet")  # atomic: the result file is either complete or absent
    os.sync()

    t = {r["metric"]: r["value"] for r in rows if r["part"] == "test"}
    print(f"{exp_id}: best epoch {best_ep:2d} | test acc {t['acc']:.3f} "
          f"macro-F1 {t['macro_f1']:.3f} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
