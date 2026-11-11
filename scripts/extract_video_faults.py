"""Fault plan, step 4 (docs/plan_faults_2026-11-11.md, section 5, option A): video faults made on the FRAMES.

Every video is decoded once; from the same decoded frames this script builds the clean clip and every planned
fault version, and runs each through the SAME frozen R(2+1)D-18 as the journal features. Everything that makes the
clean clip (frame sampling, 15 s window on MMAct, whole-frame resize, colour normalisation, network, bf16) is
IMPORTED from extract_video_features_v2.py, which is not edited.

  f_gap    within the 32 sampled frames, one stretch of round(share x 32) frames = 3 / 8 / 16 (share 10 / 25 / 50 %)
           at a random position is frozen on the frame before it (a gap at the very start: the frame after it).
           The same rule as the sensor gap (trustmi.faults), on the video's own samples.
  f_shift  MMAct only: the frames shown come from 0.2 / 0.5 / 1.0 s earlier or later (random direction) on the video's
           own timeline, at its own frame rate, kept inside the clip's window (the vacated end holds the edge frame),
           then the same 32 positions are sampled. The same rule as the sensor shift.
  Random draws are keyed on (fault, level, clip): all camera views of one MMAct clip get the SAME gap or shift, so
  the whole video sensor fails at once, as a watch or phone does. The keys use the same fault ids as trustmi.faults.

CHECK: the clean version made here is compared with the stored journal features (the loaders' video input).
Recomputing on the GPU can differ in the last digits (bf16, different batch shapes); the size of that difference is
printed next to the size of each fault's effect and saved in the .json, and the test bench refuses fault features
whose clean version is off by more than VideoFaultBank.CLEAN_TOL.

  python scripts/extract_video_faults.py --dataset utd                  UTD-MHAD (gap only), about a minute
  python scripts/extract_video_faults.py --dataset mmact --subjects 1   timing run, one subject (kept as a shard)
  python scripts/extract_video_faults.py --dataset mmact                all subjects, resumable (finished shards skipped)
Output: data/features/video_r2plus1d18_v2_faults.npz, data/features/mmact_video_r2plus1d18_v2_faults.npz (+ .json)
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import extract_video_features_v2 as V                  # the journal feature rules, unchanged
from trustmi.faults import FAULTS, FAULT_SALT, GAP, SHIFT_S, VIDEO_FAULTS

T = V.T
UTD_FAULTS = [("f_gap", lv) for lv in (1, 2, 3)]
MM_FAULTS = UTD_FAULTS + [("f_shift", lv) for lv in (1, 2, 3)]
MM_SHARDS = Path("data/features/mmact_v2_faults_shards")


def key(fault, lv):
    return f"{fault}_L{lv}"


def clip_rng(fault, lv, trial):
    h = int(hashlib.sha256(str(trial).encode()).hexdigest()[:12], 16)
    return np.random.default_rng([0, FAULT_SALT + FAULTS.index(fault), lv, h])


def gap_map(lv, trial):
    """Positions 0..T-1 of the sampled frames after the gap (as trustmi.faults.FaultBank._gap, with n = T)."""
    g = min(max(1, int(round(GAP[lv] * T))), T - 1)
    a = int(clip_rng("f_gap", lv, trial).integers(0, T - g + 1))
    m = np.arange(T)
    m[a:a + g] = a - 1 if a > 0 else a + g
    return m, a


def shift_dir(lv, trial):
    return int(clip_rng("f_shift", lv, trial).choice([-1, 1]))       # +1 later (delay), -1 earlier


class FaultVideos(Dataset):
    """One item = one video -> (n_versions, 3, T, H, W): clean first, then the faults in `faults` order."""

    def __init__(self, paths, trials, window_s, faults):
        self.paths, self.trials, self.window_s, self.faults = list(paths), list(trials), window_s, faults

    def _plan(self, n, fps, trial):
        a, b, idx = V._pick(n, fps, self.window_s)
        out = [idx]
        for fault, lv in self.faults:
            if fault == "f_gap":
                out.append(idx[gap_map(lv, trial)[0]])
            else:
                k = int(round(SHIFT_S[lv] * fps))
                out.append(np.clip(idx - shift_dir(lv, trial) * k, a, b - 1))
        return out

    def _read(self, path, trial):
        cap = V._open(path)
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 0:
            plan = self._plan(n, fps, trial)
            need, got = set(np.concatenate(plan).tolist()), {}
            ok_all = True
            for i in range(max(need) + 1):
                if not cap.grab():
                    ok_all = False; break
                if i in need:
                    ok, f = cap.retrieve()
                    if not ok:
                        ok_all = False; break
                    got[i] = V._shrink(f)
            cap.release()
            if ok_all:
                return plan, got, n, fps
        else:
            cap.release()
        cap = V._open(path)                                   # fallback: decode every frame (as read_clip_full)
        fps, frames = cap.get(cv2.CAP_PROP_FPS), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(V._shrink(f))
        cap.release()
        if not frames:
            raise RuntimeError(f"{path}: no decodable frames")
        n = len(frames)
        plan = self._plan(n, fps, trial)
        return plan, {i: frames[i] for i in set(np.concatenate(plan).tolist())}, n, fps

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        plan, got, n, fps = self._read(self.paths[i], self.trials[i])
        clips = [np.stack([got[j] for j in p])[..., ::-1] for p in plan]       # BGR -> RGB, (T, H, W, 3)
        x = torch.from_numpy(np.ascontiguousarray(np.stack(clips))).permute(0, 4, 1, 2, 3).float().div_(255)
        return (x - V.MEAN) / V.STD, n, fps                                   # (n_versions, 3, T, H, W)


@torch.no_grad()
def embed_all(model, paths, trials, window_s, faults, workers, batch):
    dl = DataLoader(FaultVideos(paths, trials, window_s, faults), batch_size=batch, num_workers=workers,
                    worker_init_fn=V._worker_init, pin_memory=True)
    nv, tok, info = 1 + len(faults), [], []
    t0, done, next_report = time.time(), 0, 200
    for x, n, fps in dl:
        B = x.shape[0]
        done += B
        if done >= next_report or done == len(paths):
            rate = done / (time.time() - t0)
            print(f"    {done}/{len(paths)} videos | {rate:.1f} videos/s | "
                  f"~{(len(paths) - done) / rate / 60:.1f} min left in this batch", flush=True)
            next_report += 200
        with torch.autocast("cuda", dtype=torch.bfloat16):                 # exactly as extract_video_features_v2
            y = model.stem(x.flatten(0, 1).cuda(non_blocking=True))
            y = model.layer4(model.layer3(model.layer2(model.layer1(y))))
            t = y.mean(dim=(3, 4)).transpose(1, 2)                          # (B * nv, T', 512)
        tok.append(t.float().cpu().numpy().reshape(B, nv, *t.shape[1:]))
        info.append(np.stack([n.numpy(), fps.numpy()], axis=1))
    tok = np.concatenate(tok)                                               # (videos, nv, 4, 512)
    assert len(tok) == len(paths) and np.isfinite(tok).all()
    return tok, np.concatenate(info)


def report(stored, versions, names, label):
    """How far the recomputed clean is from the stored features, next to the size of each fault's effect."""
    noise = np.abs(versions["clean"] - stored)
    sd = stored.std(axis=(0, 1)) + 1e-6                                     # per feature, as the loaders normalise
    rows = {"clean recomputed vs stored": noise / sd}
    for k in names:
        rows[f"{k} vs clean"] = np.abs(versions[k] - versions["clean"]) / sd
    print(f"\n{label}: mean | max absolute difference, in units of each feature's spread")
    summ = {}
    for r, v in rows.items():
        summ[r] = dict(mean=float(v.mean()), p999=float(np.quantile(v, 0.999)), max=float(v.max()))
        print(f"  {r:28s} mean {v.mean():.4f} | 99.9% {np.quantile(v, 0.999):.4f} | max {v.max():.4f}")
    return summ


def write(out, trial, versions, extra):
    V.save_atomic(out, trial=np.asarray(trial), **{k: v.astype(np.float32) for k, v in versions.items()},
                  **extra.pop("arrays", {}))
    meta = dict(plan="docs/plan_faults_2026-11-11.md (section 5, option A)", base="extract_video_features_v2.py",
                frames_per_clip=T, gap_frames={lv: min(max(1, int(round(GAP[lv] * T))), T - 1) for lv in GAP},
                shift_s=SHIFT_S, torch=torch.__version__, opencv=cv2.__version__, **extra)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"-> {out}")


def run_utd(model, args):
    trials = np.load(V.UTD_META, allow_pickle=True)["trial"]
    t0 = time.time()
    tok, info = embed_all(model, [V.UTD_ROOT / f"{t}_color.avi" for t in trials], trials, None, UTD_FAULTS,
                          args.workers, args.batch)
    names = [key(*f) for f in UTD_FAULTS]
    versions = {"clean": tok[:, 0], **{k: tok[:, j + 1] for j, k in enumerate(names)}}
    stored = np.load(V.UTD_OUT)
    if not (stored["trial"] == trials).all():
        raise SystemExit("stored UTD-MHAD features are in a different trial order: stop and paste this")
    print(f"UTD-MHAD: {len(trials)} videos in {time.time() - t0:.0f} s")
    summ = report(stored["tokens"].astype(np.float32), versions, names, "UTD-MHAD")
    gaps = np.array([[gap_map(lv, t)[1] for lv in (1, 2, 3)] for t in trials])
    write(VIDEO_FAULTS["d1v2"], trials, versions,
          dict(dataset="UTD-MHAD", faults=names, check=summ, arrays=dict(gap_start=gaps)))


def run_mmact(model, args):
    clips = pd.read_csv(V.MM_CLIPS)
    clips = clips[clips.status3.isin(V.MM_KEEP)]
    vids = pd.read_csv(V.MM_INDEX).merge(clips[V.KEY + ["status3"]], on=V.KEY)
    vids["trial"] = [f"s{a:02d}_sc{b}_se{c}_{d}" for a, b, c, d in vids[V.KEY].itertuples(index=False)]
    names = [key(*f) for f in MM_FAULTS]
    MM_SHARDS.mkdir(parents=True, exist_ok=True)
    for s in args.subjects or sorted(vids.subject.unique()):
        shard = MM_SHARDS / f"subject{s:02d}.npz"
        if shard.exists():
            print(f"subject {s:2d}: already done, skipped"); continue
        v = vids[vids.subject == s]
        t0 = time.time()
        tok, info = embed_all(model, [V.MM_ROOT / p for p in v.path], v.trial, V.MM_WINDOW_S, MM_FAULTS,
                              args.workers, args.batch)
        V.save_atomic(shard, path=v.path.to_numpy().astype(str), tokens=tok.astype(np.float16),
                      n_frames=info[:, 0], fps=info[:, 1])
        dt = time.time() - t0
        print(f"subject {s:2d}: {len(v)} videos in {dt:.0f} s ({len(v) / dt:.1f} videos/s)", flush=True)

    done = sorted(MM_SHARDS.glob("subject*.npz"))
    if len(done) < vids.subject.nunique():
        print(f"{len(done)}/{vids.subject.nunique()} subjects done; merged file not written yet"); return

    # merge exactly as extract_video_features_v2: per-clip mean over the available views, same clip order
    parts = [np.load(p) for p in done]
    view = pd.DataFrame(dict(path=np.concatenate([p["path"] for p in parts])))
    tok = np.concatenate([p["tokens"] for p in parts]).astype(np.float32)        # (views, nv, 4, 512)
    view = view.merge(vids[["path"] + V.KEY], on="path", how="left", validate="1:1")
    assert len(view) == len(vids) and view.subject.notna().all()
    view["row"] = np.arange(len(view))
    clips = clips.sort_values(V.KEY).reset_index(drop=True)
    g = {k: r.row.to_numpy() for k, r in view.groupby(V.KEY)}
    rows = [g[tuple(k)] for k in clips[V.KEY].itertuples(index=False)]
    trial = np.array([f"s{a:02d}_sc{b}_se{c}_{d}" for a, b, c, d in clips[V.KEY].itertuples(index=False)])
    per = np.stack([tok[r].mean(0) for r in rows])                                # (clips, nv, 4, 512)
    versions = {"clean": per[:, 0], **{k: per[:, j + 1] for j, k in enumerate(names)}}
    stored = np.load(V.MM_OUT)
    if not (stored["trial"] == trial).all():
        raise SystemExit("stored MMAct features are in a different clip order: stop and paste this")
    summ = report(stored["tokens"].astype(np.float32), versions, names, "MMAct")
    gaps = np.array([[gap_map(lv, t)[1] for lv in (1, 2, 3)] for t in trial])
    dirs = np.array([[shift_dir(lv, t) for lv in (1, 2, 3)] for t in trial])
    write(VIDEO_FAULTS["d2v"], trial, versions,
          dict(dataset="MMAct V1 trimmed", faults=names, clips=len(trial), views=len(view), check=summ,
               clip_feature="mean over available camera views; one fault draw per clip, shared by its views",
               arrays=dict(gap_start=gaps, shift_dir=dirs, n_views=np.array([len(r) for r in rows]))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["utd", "mmact"], required=True)
    ap.add_argument("--subjects", type=int, nargs="*", help="MMAct only: subset of subjects (timing run)")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--batch", type=int, default=None, help="videos per batch (default: utd 4, mmact 2)")
    args = ap.parse_args()
    args.batch = args.batch or (4 if args.dataset == "utd" else 2)
    torch.backends.cudnn.benchmark = False
    model = V.r2plus1d_18(weights=V.WEIGHTS).eval().cuda()
    (run_utd if args.dataset == "utd" else run_mmact)(model, args)
