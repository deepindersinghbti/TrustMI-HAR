"""Addition 1 (docs/plan_additions_2026-11-14.md, 1.2): video stuck and over-exposure, made on the FRAMES.

Exactly the method of scripts/extract_video_faults.py (imported, NOT edited): every video is decoded once; from the
same decoded frames the clean clip and every fault version are built and run through the SAME frozen R(2+1)D-18 as
the journal features (frame sampling, 15 s MMAct window, resize, colour normalisation, network, bf16 all imported).
  f_stuck  within the 32 sampled frames, the camera freezes and never recovers: from the onset to the last frame
           the onset frame is repeated; the frozen stretch is 8 / 16 / 32 frames (25 / 50 / 100 %), so the onset is
           frame 24 / 16 / 0 (the same rule as the sensor stuck fault, trustmi.faults2.stuck_span, with n = 32).
  f_sat    over-exposure: every sampled frame's pixel values multiplied by 1.5 / 2 / 3 and clipped at 255, before
           the colour normalisation. The same frames as the clean clip.
Both are deterministic: no random draw. All camera views of an MMAct clip get the same fault (the whole video
sensor fails at once, as in extract_video_faults.py).
CHECK: the clean version made here is compared with the stored journal features, as in extract_video_faults.py;
the test bench (trustmi.faults2.VideoFaultBank2) refuses features whose clean version is off by more than CLEAN_TOL.

  python scripts/extract_video_faults2.py --dataset utd                    UTD-MHAD, about a minute
  python scripts/extract_video_faults2.py --dataset mmact --subjects 1     timing run, one subject (kept as a shard)
  python scripts/extract_video_faults2.py --dataset mmact                  all subjects, resumable
Output: data/features/video_r2plus1d18_v2_faults2.npz, data/features/mmact_video_r2plus1d18_v2_faults2.npz (+ .json)
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import extract_video_faults as EF                     # the frame-level fault method, unchanged
from trustmi.faults2 import STUCK, VIDEO_GAIN, VIDEO_FAULTS2, stuck_span

V, T = EF.V, EF.T
FAULTS2 = [("f_stuck", lv) for lv in (1, 2, 3)] + [("f_sat", lv) for lv in (1, 2, 3)]
MM_SHARDS = ROOT / "data/features/mmact_v2_faults2_shards"
key = EF.key


def stuck_map(lv):
    """Positions 0..T-1 of the sampled frames after the camera freezes (trustmi.faults2.stuck_span, n = T)."""
    a, _ = stuck_span(STUCK[lv], T)
    m = np.arange(T)
    m[a:] = a
    return m, a


class FaultVideos2(EF.FaultVideos):
    """As EF.FaultVideos (same decoding, same clean clip); the versions are the stuck and over-exposure faults."""

    def _plan(self, n, fps, trial):
        a, b, idx = V._pick(n, fps, self.window_s)
        return [idx] + [idx[stuck_map(lv)[0]] if fault == "f_stuck" else idx for fault, lv in self.faults]

    def __getitem__(self, i):
        plan, got, n, fps = self._read(self.paths[i], self.trials[i])
        clips = [np.stack([got[j] for j in p])[..., ::-1].astype(np.float32) for p in plan]   # RGB, (T, H, W, 3)
        for j, (fault, lv) in enumerate(self.faults, start=1):
            if fault == "f_sat":
                clips[j] = np.minimum(clips[j] * VIDEO_GAIN[lv], 255.0)          # over-exposure, pixel range
        x = torch.from_numpy(np.ascontiguousarray(np.stack(clips))).permute(0, 4, 1, 2, 3).div_(255)
        return (x - V.MEAN) / V.STD, n, fps                                     # (n_versions, 3, T, H, W)


@torch.no_grad()
def embed_all(model, paths, trials, window_s, workers, batch):
    """EF.embed_all with FaultVideos2 (the network part is identical)."""
    dl = DataLoader(FaultVideos2(paths, trials, window_s, FAULTS2), batch_size=batch, num_workers=workers,
                    worker_init_fn=V._worker_init, pin_memory=True)
    nv, tok, info = 1 + len(FAULTS2), [], []
    t0, done, next_report = time.time(), 0, 200
    for x, n, fps in dl:
        B = x.shape[0]
        done += B
        if done >= next_report or done == len(paths):
            rate = done / (time.time() - t0)
            print(f"    {done}/{len(paths)} videos | {rate:.1f} videos/s | "
                  f"~{(len(paths) - done) / rate / 60:.1f} min left in this batch", flush=True)
            next_report += 200
        with torch.autocast("cuda", dtype=torch.bfloat16):                   # exactly as extract_video_features_v2
            y = model.stem(x.flatten(0, 1).cuda(non_blocking=True))
            y = model.layer4(model.layer3(model.layer2(model.layer1(y))))
            t = y.mean(dim=(3, 4)).transpose(1, 2)
        tok.append(t.float().cpu().numpy().reshape(B, nv, *t.shape[1:]))
        info.append(np.stack([n.numpy(), fps.numpy()], axis=1))
    tok = np.concatenate(tok)
    assert len(tok) == len(paths) and np.isfinite(tok).all()
    return tok, np.concatenate(info)


def write(out, trial, versions, extra):
    V.save_atomic(out, trial=np.asarray(trial).astype(str), **{k: v.astype(np.float32) for k, v in versions.items()})
    meta = dict(plan="docs/plan_additions_2026-11-14.md (Addition 1.2)", base="extract_video_faults.py",
                frames_per_clip=T, stuck_onset_frame={lv: stuck_map(lv)[1] for lv in STUCK}, gain=VIDEO_GAIN,
                torch=torch.__version__, opencv=cv2.__version__, **extra)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"-> {out}")


def run_utd(model, args):
    trials = np.load(V.UTD_META, allow_pickle=True)["trial"].astype(str)
    stored = np.load(V.UTD_OUT, allow_pickle=True)
    if not (stored["trial"].astype(str) == trials).all():
        raise SystemExit("stored UTD-MHAD features are in a different trial order: stop and paste this")
    t0 = time.time()
    tok, _ = embed_all(model, [V.UTD_ROOT / f"{t}_color.avi" for t in trials], trials, None, args.workers, args.batch)
    names = [key(*f) for f in FAULTS2]
    versions = {"clean": tok[:, 0], **{k: tok[:, j + 1] for j, k in enumerate(names)}}
    print(f"UTD-MHAD: {len(trials)} videos in {time.time() - t0:.0f} s")
    summ = EF.report(stored["tokens"].astype(np.float32), versions, names, "UTD-MHAD")
    write(VIDEO_FAULTS2["d1v2"], trials, versions, dict(dataset="UTD-MHAD", faults=names, check=summ))


def run_mmact(model, args):
    clips = pd.read_csv(V.MM_CLIPS)
    clips = clips[clips.status3.isin(V.MM_KEEP)]
    vids = pd.read_csv(V.MM_INDEX).merge(clips[V.KEY + ["status3"]], on=V.KEY)
    vids["trial"] = [f"s{a:02d}_sc{b}_se{c}_{d}" for a, b, c, d in vids[V.KEY].itertuples(index=False)]
    names = [key(*f) for f in FAULTS2]
    order = clips.sort_values(V.KEY)
    expect = np.array([f"s{a:02d}_sc{b}_se{c}_{d}" for a, b, c, d in order[V.KEY].itertuples(index=False)])
    stored = np.load(V.MM_OUT, allow_pickle=True)
    if not (stored["trial"].astype(str) == expect).all():
        raise SystemExit("stored MMAct features are in a different clip order: stop and paste this")
    MM_SHARDS.mkdir(parents=True, exist_ok=True)
    for s in args.subjects or sorted(vids.subject.unique()):
        shard = MM_SHARDS / f"subject{s:02d}.npz"
        if shard.exists():
            print(f"subject {s:2d}: already done, skipped"); continue
        v = vids[vids.subject == s]
        t0 = time.time()
        tok, info = embed_all(model, [V.MM_ROOT / p for p in v.path], v.trial, V.MM_WINDOW_S, args.workers, args.batch)
        V.save_atomic(shard, path=v.path.to_numpy().astype(str), tokens=tok.astype(np.float16),
                      n_frames=info[:, 0], fps=info[:, 1])
        dt = time.time() - t0
        print(f"subject {s:2d}: {len(v)} videos in {dt:.0f} s ({len(v) / dt:.1f} videos/s)", flush=True)
    done = sorted(MM_SHARDS.glob("subject*.npz"))
    if len(done) < vids.subject.nunique():
        print(f"{len(done)}/{vids.subject.nunique()} subjects done; merged file not written yet"); return
    # merge exactly as extract_video_faults.py: per-clip mean over the available views, same clip order
    parts = [np.load(p) for p in done]
    view = pd.DataFrame(dict(path=np.concatenate([p["path"] for p in parts])))
    tok = np.concatenate([p["tokens"] for p in parts]).astype(np.float32)
    view = view.merge(vids[["path"] + V.KEY], on="path", how="left", validate="1:1")
    assert len(view) == len(vids) and view.subject.notna().all()
    view["row"] = np.arange(len(view))
    clips = clips.sort_values(V.KEY).reset_index(drop=True)
    g = {k: r.row.to_numpy() for k, r in view.groupby(V.KEY)}
    rows = [g[tuple(k)] for k in clips[V.KEY].itertuples(index=False)]
    trial = np.array([f"s{a:02d}_sc{b}_se{c}_{d}" for a, b, c, d in clips[V.KEY].itertuples(index=False)])
    per = np.stack([tok[r].mean(0) for r in rows])
    versions = {"clean": per[:, 0], **{k: per[:, j + 1] for j, k in enumerate(names)}}
    assert (trial == expect).all()
    summ = EF.report(stored["tokens"].astype(np.float32), versions, names, "MMAct")
    write(VIDEO_FAULTS2["d2v"], trial, versions,
          dict(dataset="MMAct V1 trimmed", faults=names, clips=len(trial), views=len(view), check=summ,
               clip_feature="mean over available camera views; the same fault for all views of a clip"))


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
