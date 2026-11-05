"""Video features v2 (journal version): frozen R(2+1)D-18 on the WHOLE frame.

One rule for every dataset (docs/data_audit_mmact_video.md, decisions 3-5 and 9):
  - decode every frame, shrink it to height 112 with the aspect ratio kept (no crop; cv2.INTER_AREA)
      MMAct 1920x1080 -> 112x199, UTD-MHAD 640x480 -> 112x149
  - 32 frames evenly spaced over the clip (frames repeat if a clip has fewer than 32)
  - MMAct only: a video longer than 15 s uses its centred 15 s, the same window the sensor loader uses
  - colour normalisation = mean/std of the model's own preset (weights.transforms())
  - frozen torchvision r2plus1d_18 (Kinetics-400): 4 time-step tokens (512 each) + pooled 512 per video

Usage (from the project root):
  python scripts/extract_video_features_v2.py --dataset utd
  python scripts/extract_video_features_v2.py --dataset mmact --subjects 1     # timing run, one subject
  python scripts/extract_video_features_v2.py --dataset mmact                  # all subjects, resumable
MMAct is saved one subject at a time (data/features/mmact_v2_shards/); finished subjects are skipped,
so the run can simply be started again after any stop. The merged file is written once all 20 exist.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

T, H = 32, 112
WEIGHTS = R2Plus1D_18_Weights.KINETICS400_V1
PREP = WEIGHTS.transforms()
MEAN = torch.tensor(PREP.mean).view(3, 1, 1, 1)
STD = torch.tensor(PREP.std).view(3, 1, 1, 1)

UTD_ROOT = Path("data/extracted/UTD-MHAD/RGB/RGB")
UTD_META = "data/processed/utd_mhad_v1.npz"
UTD_OUT = Path("data/features/video_r2plus1d18_v2.npz")
MM_ROOT = Path("data/extracted/MMAct")
MM_INDEX = "data/processed/mmact_video_index.csv"
MM_CLIPS = "data/processed/mmact_video_clips.csv"
MM_SHARDS = Path("data/features/mmact_v2_shards")
MM_OUT = Path("data/features/mmact_video_r2plus1d18_v2.npz")
MM_WINDOW_S = 15            # = WINDOW_S in src/trustmi/data/mmact.py
MM_KEEP = ("main3", "no_watch3")
KEY = ["subject", "scene", "session", "action"]


def _pick(n, fps, window_s):
    """Frame window [a, b) (centred window_s seconds if the clip is longer) and the T evenly spaced frames."""
    a, b = 0, n
    if window_s is not None and fps > 0 and n > round(window_s * fps):
        m = round(window_s * fps)
        a = (n - m) // 2
        b = a + m
    return a, b, np.linspace(a, b - 1, T).round().astype(int)


def _shrink(f):
    h, w = f.shape[:2]
    return cv2.resize(f, (round(H * w / h), H), interpolation=cv2.INTER_AREA)


def _open(path):
    """One decoder thread per video; parallelism comes from the DataLoader workers."""
    try:
        return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    except (AttributeError, cv2.error):
        return cv2.VideoCapture(str(path))


def read_clip_full(path, window_s=None):
    """Reference method: decode, convert and shrink EVERY frame, then pick T of them."""
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(_shrink(f))
    cap.release()
    n = len(frames)
    if n == 0:
        raise RuntimeError(f"{path}: no decodable frames")
    a, b, idx = _pick(n, fps, window_s)
    clip = np.ascontiguousarray(np.stack([frames[i] for i in idx])[..., ::-1])   # BGR -> RGB, (T, H, W, 3)
    return clip, n, a, b


def read_clip(path, window_s=None):
    """Fast method, same frames and pixels as read_clip_full: plan the T frames from the header frame count,
    decode (grab) up to the last one, and convert + shrink only the chosen frames. Falls back to
    read_clip_full if the video ends before its header says."""
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return read_clip_full(path, window_s)
    a, b, idx = _pick(n, fps, window_s)
    need, got = set(idx.tolist()), {}
    for i in range(int(idx[-1]) + 1):
        if not cap.grab():
            cap.release()
            return read_clip_full(path, window_s)
        if i in need:
            ok, f = cap.retrieve()
            if not ok:
                cap.release()
                return read_clip_full(path, window_s)
            got[i] = _shrink(f)
    cap.release()
    clip = np.ascontiguousarray(np.stack([got[i] for i in idx])[..., ::-1])
    return clip, n, a, b


class Videos(Dataset):
    def __init__(self, paths, window_s):
        self.paths, self.window_s = list(paths), window_s

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        clip, n, a, b = read_clip(self.paths[i], self.window_s)
        x = torch.from_numpy(clip).permute(3, 0, 1, 2).float().div_(255)          # (3, T, H, W)
        return (x - MEAN) / STD, n, a, b


def _worker_init(_):
    cv2.setNumThreads(1)       # one OpenCV thread per worker (resize); decoder threads are set in _open()


@torch.no_grad()
def embed_all(model, paths, window_s, workers, batch):
    dl = DataLoader(Videos(paths, window_s), batch_size=batch, num_workers=workers,
                    worker_init_fn=_worker_init, pin_memory=True)
    tok, pool, info = [], [], []
    t0, done, next_report = time.time(), 0, 200
    for x, n, a, b in dl:
        done += len(n)
        if done >= next_report or done == len(paths):
            rate = done / (time.time() - t0)
            print(f"    {done}/{len(paths)} videos | {rate:.1f} videos/s | "
                  f"~{(len(paths) - done) / rate / 60:.1f} min left in this batch", flush=True)
            next_report += 200
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model.stem(x.cuda(non_blocking=True))
            y = model.layer4(model.layer3(model.layer2(model.layer1(y))))          # (B, 512, T', h, w)
            t = y.mean(dim=(3, 4)).transpose(1, 2)                                  # (B, T', 512)
        tok.append(t.float().cpu().numpy())
        pool.append(t.float().mean(dim=1).cpu().numpy())
        info.append(np.stack([n.numpy(), a.numpy(), b.numpy()], axis=1))
    tok, pool, info = np.concatenate(tok), np.concatenate(pool), np.concatenate(info)
    assert len(tok) == len(paths) and np.isfinite(tok).all()
    return tok, pool, info


def save_atomic(out, **arrays):
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)


def write_json(path, extra):
    meta = dict(model="torchvision r2plus1d_18", weights="KINETICS400_V1", frozen=True,
                frames_per_clip=T, sampling="evenly spaced over each clip (np.linspace + round)",
                input=f"whole frame, no crop, aspect kept, height {H} (cv2.INTER_AREA)",
                normalisation=dict(mean=list(PREP.mean), std=list(PREP.std)),
                decisions="docs/data_audit_mmact_video.md", torch=torch.__version__,
                opencv=cv2.__version__, **extra)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def run_utd(model, args):
    trials = np.load(UTD_META, allow_pickle=True)["trial"]
    t0 = time.time()
    tok, pool, info = embed_all(model, [UTD_ROOT / f"{t}_color.avi" for t in trials], None,
                                args.workers, args.batch)
    save_atomic(UTD_OUT, tokens=tok, pooled=pool, trial=trials, decoded_frames=info[:, 0])
    write_json(UTD_OUT, dict(dataset="UTD-MHAD", window="whole clip", tokens_shape=list(tok.shape),
                             replaces="video_r2plus1d18_v1 (centre crop 112x112) for the journal version"))
    print(f"UTD-MHAD: {len(trials)} videos in {time.time() - t0:.0f} s | tokens {tok.shape} | "
          f"decoded frames min/max {info[:, 0].min()}/{info[:, 0].max()} -> {UTD_OUT}")


def run_mmact(model, args):
    clips = pd.read_csv(MM_CLIPS)
    clips = clips[clips.status3.isin(MM_KEEP)]
    vids = pd.read_csv(MM_INDEX).merge(clips[KEY + ["status3"]], on=KEY)
    MM_SHARDS.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or sorted(vids.subject.unique())
    for s in subjects:
        shard = MM_SHARDS / f"subject{s:02d}.npz"
        if shard.exists():
            print(f"subject {s:2d}: already done, skipped")
            continue
        v = vids[vids.subject == s]
        t0 = time.time()
        tok, pool, info = embed_all(model, [MM_ROOT / p for p in v.path], MM_WINDOW_S, args.workers, args.batch)
        save_atomic(shard, path=v.path.to_numpy().astype(str), tokens=tok.astype(np.float16),
                    pooled=pool.astype(np.float16), n_frames=info[:, 0], win_start=info[:, 1], win_end=info[:, 2])
        dt = time.time() - t0
        print(f"subject {s:2d}: {len(v)} videos in {dt:.0f} s ({len(v) / dt:.1f} videos/s) | "
              f"windowed (>15 s): {(info[:, 1] > 0).sum()}", flush=True)

    done = sorted(MM_SHARDS.glob("subject*.npz"))
    if len(done) < vids.subject.nunique():
        print(f"{len(done)}/{vids.subject.nunique()} subjects done; merged file not written yet")
        return

    # merge: per-view arrays + per-clip mean over the available views (decision 1)
    parts = [np.load(p) for p in done]
    view = pd.DataFrame(dict(path=np.concatenate([p["path"] for p in parts])))
    tok = np.concatenate([p["tokens"] for p in parts]).astype(np.float32)
    pool = np.concatenate([p["pooled"] for p in parts]).astype(np.float32)
    view = view.merge(vids[["path", "cam"] + KEY], on="path", how="left", validate="1:1")
    assert view.cam.notna().all() and len(view) == len(vids)
    view["row"] = np.arange(len(view))
    clips = clips.sort_values(KEY).reset_index(drop=True)
    g = {k: r.row.to_numpy() for k, r in view.groupby(KEY)}
    rows = [g[tuple(k)] for k in clips[KEY].itertuples(index=False)]
    trial = [f"s{a:02d}_sc{b}_se{c}_{d}" for a, b, c, d in clips[KEY].itertuples(index=False)]
    save_atomic(MM_OUT, trial=np.array(trial), status3=clips.status3.to_numpy().astype(str),
                tokens=np.stack([tok[r].mean(0) for r in rows]),
                pooled=np.stack([pool[r].mean(0) for r in rows]),
                n_views=np.array([len(r) for r in rows]),
                view_path=view.path.to_numpy().astype(str), view_cam=view.cam.to_numpy().astype(int),
                view_trial=np.array([f"s{a:02d}_sc{b}_se{c}_{d}" for a, b, c, d in view[KEY].itertuples(index=False)]),
                view_tokens=tok.astype(np.float16), view_pooled=pool.astype(np.float16),
                view_n_frames=np.concatenate([p["n_frames"] for p in parts]),
                view_win_start=np.concatenate([p["win_start"] for p in parts]),
                view_win_end=np.concatenate([p["win_end"] for p in parts]))
    write_json(MM_OUT, dict(dataset="MMAct V1 trimmed", clips=len(clips), views=len(view),
                            window=f"centred {MM_WINDOW_S} s if longer (as the sensor loader)",
                            clip_feature="mean over available camera views",
                            clip_status=sorted(MM_KEEP), tokens_shape=[len(clips), 4, 512]))
    print(f"merged: {len(clips)} clips ({clips.status3.value_counts().to_dict()}) from {len(view)} views -> {MM_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["utd", "mmact"], required=True)
    ap.add_argument("--subjects", type=int, nargs="*", help="MMAct only: subset of subjects (timing run)")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    model = r2plus1d_18(weights=WEIGHTS).eval().cuda()
    (run_utd if args.dataset == "utd" else run_mmact)(model, args)
