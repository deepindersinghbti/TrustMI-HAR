"""Compare re-trained runs with their backed-up originals (the 16 WearGait-PD runs that started while code was
uncommitted on 12 Nov). For each run id, compares:
  weights      runs/<id>/ckpt_best.pt, every tensor, exactly
  curve        runs/<id>/log.csv, every column except timings
  predictions  runs/<id>/preds_*.parquet (classifier runs only), exactly
  scores       results/raw/<id>.parquet, every column except git_sha
    python scripts/compare_reruns.py runs/dirty_flag_check <id> [<id> ...]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def tensors(obj, prefix=""):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from tensors(obj[k], f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from tensors(v, f"{prefix}[{i}]")


def same_weights(a, b):
    ta = dict(tensors(torch.load(a, map_location="cpu", weights_only=False)))
    tb = dict(tensors(torch.load(b, map_location="cpu", weights_only=False)))
    return ta.keys() == tb.keys() and all(torch.equal(ta[k], tb[k]) for k in ta), len(ta)


def same_frame(a, b, drop=()):
    fa, fb = a.drop(columns=[c for c in a if c in drop]), b.drop(columns=[c for c in b if c in drop])
    return fa.equals(fb)


def main():
    backup, ids = Path(sys.argv[1]), sys.argv[2:]
    bad = 0
    for rid in ids:
        old, new = backup / rid, ROOT / "runs" / rid
        res = {}
        ok_w, n = same_weights(old / "ckpt_best.pt", new / "ckpt_best.pt")
        res[f"weights ({n} tensors)"] = ok_w
        la, lb = pd.read_csv(old / "log.csv"), pd.read_csv(new / "log.csv")
        timing = [c for c in la if any(t in c.lower() for t in ("time", "sec", "elapsed"))]
        res["curve"] = same_frame(la, lb, timing)
        for p in sorted(old.glob("preds_*.parquet")):
            res[p.stem] = same_frame(pd.read_parquet(p), pd.read_parquet(new / p.name))
        res["scores"] = same_frame(pd.read_parquet(backup / f"{rid}.parquet"),
                                   pd.read_parquet(ROOT / "results/raw" / f"{rid}.parquet"), ("git_sha",))
        ok = all(res.values())
        bad += not ok
        print(f"{rid:32s} {'IDENTICAL' if ok else 'DIFFERENT'}   "
              + "  ".join(f"{k} {'ok' if v else 'DIFFERS'}" for k, v in res.items()))
    print(f"\n{len(ids) - bad} of {len(ids)} runs identical")


if __name__ == "__main__":
    main()
