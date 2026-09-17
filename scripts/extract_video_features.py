import cv2, json, numpy as np, torch
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

ROOT = "data/extracted/UTD-MHAD/RGB/RGB"
OUT = "data/features/video_r2plus1d18_v1.npz"
T = 32
BATCH = 8

meta = np.load("data/processed/utd_mhad_v1.npz", allow_pickle=True)
trials = meta["trial"]

weights = R2Plus1D_18_Weights.KINETICS400_V1
model = r2plus1d_18(weights=weights).eval().cuda()
prep = weights.transforms()          # expects (T, C, H, W) uint8, returns (C, T, H, W)

def load_clip(tid):
    cap = cv2.VideoCapture(f"{ROOT}/{tid}_color.avi")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    n = len(frames)
    assert n >= T, f"{tid}: only {n} decodable frames"
    idx = np.linspace(0, n - 1, T).round().astype(int)
    clip = torch.from_numpy(np.stack([frames[i] for i in idx]))   # (T, H, W, 3)
    return prep(clip.permute(0, 3, 1, 2)), n                         # (3, T, 112, 112)

@torch.no_grad()
def embed(x):
    x = model.stem(x)
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))  # (B, 512, T', h, w)
    tokens = x.mean(dim=(3, 4)).transpose(1, 2)                     # (B, T', 512)
    return tokens, tokens.mean(dim=1)                               # pooled (B, 512)

all_tok, all_pool, decoded = [], [], []
for i in range(0, len(trials), BATCH):
    batch = [load_clip(t) for t in trials[i:i + BATCH]]
    x = torch.stack([b[0] for b in batch]).cuda()
    decoded += [b[1] for b in batch]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tok, pool = embed(x)
    all_tok.append(tok.float().cpu().numpy())
    all_pool.append(pool.float().cpu().numpy())
    if i % 80 == 0:
        print(f"{i}/{len(trials)}", flush=True)

tokens = np.concatenate(all_tok)
pooled = np.concatenate(all_pool)
assert len(tokens) == len(trials) and np.isfinite(tokens).all()

np.savez_compressed(OUT, tokens=tokens, pooled=pooled, trial=trials,
                    decoded_frames=np.array(decoded))
with open(OUT.replace(".npz", ".json"), "w") as f:
    json.dump(dict(model="torchvision r2plus1d_18", weights="KINETICS400_V1",
                   pretraining="Kinetics-400 (YouTube); no UTD-MHAD overlap",
                   frames_per_clip=T, sampling="uniform over each clip",
                   input="112x112 after weights.transforms()",
                   tokens_shape=list(tokens.shape), pooled_shape=list(pooled.shape),
                   torch=torch.__version__), f, indent=2)

print("tokens", tokens.shape, "| pooled", pooled.shape)
print("decoded frames min/max:", min(decoded), max(decoded))
print("pooled feature norm mean:", np.linalg.norm(pooled, axis=1).mean().round(2))
