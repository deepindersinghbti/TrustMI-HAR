"""Modality encoders + cross-modal fusion transformer (B1 unimodal, B4 fusion)."""
from functools import partial
import torch
import torch.nn as nn


class IMUEnc(nn.Module):
    def __init__(self, d, c_in=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, d, 5, padding=2), nn.GELU(), nn.BatchNorm1d(d),
            nn.Conv1d(d, d, 5, padding=2, stride=2), nn.GELU(), nn.BatchNorm1d(d),
            nn.Conv1d(d, d, 5, padding=2, stride=2), nn.GELU(), nn.BatchNorm1d(d))

    def forward(self, x):                        # (B, T, C)
        return self.net(x.transpose(1, 2)).transpose(1, 2)   # (B, ceil(T/4), d)


class SkelEnc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(60, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):                        # (B, 40, 20, 3)
        return self.net(x.flatten(2))            # (B, 40, d)


class VideoEnc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):                        # (B, 4, 512)
        return self.net(x)                       # (B, 4, d)


ENCODERS = dict(video=VideoEnc, skel=SkelEnc, imu=IMUEnc,
                watch=partial(IMUEnc, c_in=3), phone=partial(IMUEnc, c_in=10))   # MMAct devices
TOKENS = dict(video=4, skel=40, imu=25, watch=188, phone=188)   # MMAct: 750 samples (15 s @ 50 Hz) -> 188
STRIDE = dict(imu=4, watch=4, phone=4)   # input samples per token, used to mask padded tokens
# journal, WearGait-PD (dataset d3; docs/plan_weargait_2026-11-11.md): 3 IMU groups, the same IMU encoder,
# 100 samples (2 s @ 50 Hz) -> 25 tokens. New entries only; existing ones unchanged.
ENCODERS.update(arms=partial(IMUEnc, c_in=12), trunk=partial(IMUEnc, c_in=18), legs=partial(IMUEnc, c_in=48))
TOKENS.update(arms=25, trunk=25, legs=25)
STRIDE.update(arms=4, trunk=4, legs=4)


class FusionTransformer(nn.Module):
    def __init__(self, modalities, n_classes=27, d=128, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.mods = list(modalities)
        self.enc = nn.ModuleDict({m: ENCODERS[m](d) for m in self.mods})
        self.type_emb = nn.ParameterDict({m: nn.Parameter(torch.zeros(1, 1, d)) for m in self.mods})
        self.pos_emb = nn.ParameterDict({m: nn.Parameter(torch.zeros(1, TOKENS[m], d)) for m in self.mods})
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        for p in [self.cls, *self.type_emb.values(), *self.pos_emb.values()]:
            nn.init.trunc_normal_(p, std=0.02)

    def _pool(self, m, t, lens):
        """Mean over tokens; with `lens`, padded tokens are excluded and flagged (True = padding)."""
        if lens is None:
            return t.mean(dim=1), None
        if m not in STRIDE:
            return t.mean(dim=1), torch.zeros(t.shape[:2], dtype=torch.bool, device=t.device)
        n_tok = torch.clamp((lens + STRIDE[m] - 1) // STRIDE[m], 1, t.shape[1])
        pad = torch.arange(t.shape[1], device=t.device)[None, :] >= n_tok[:, None]
        keep = (~pad).unsqueeze(-1).to(t.dtype)
        return (t * keep).sum(1) / keep.sum(1), pad

    def forward(self, batch, return_emb=False):
        B = batch[self.mods[0]].shape[0]
        tokens, emb, pads = [self.cls.expand(B, -1, -1)], {}, []
        lens = batch.get("len")                  # valid samples per clip (MMAct only); None for UTD-MHAD
        for m in self.mods:
            t = self.enc[m](batch[m]) + self.type_emb[m] + self.pos_emb[m]
            emb[m], pad = self._pool(m, t, lens)  # per-modality embedding (for the MI module later)
            tokens.append(t)
            pads.append(pad)
        absent = batch.get("absent")             # r_mask: {modality: (B,) bool}, True = hide from fusion
        key_pad = None
        if lens is not None or absent:
            pads = [torch.zeros(t.shape[:2], dtype=torch.bool, device=t.device) if p is None else p
                    for t, p in zip(tokens[1:], pads)]
            if absent:
                pads = [p | absent[m].to(p.device)[:, None] if m in absent else p
                        for m, p in zip(self.mods, pads)]
            key_pad = torch.cat([torch.zeros_like(pads[0][:, :1]), *pads], dim=1)
        h = self.norm(self.fusion(torch.cat(tokens, dim=1), src_key_padding_mask=key_pad))
        logits = self.head(h[:, 0])             # CLS only; emb of a hidden modality is not used
        return (logits, emb) if return_emb else logits
