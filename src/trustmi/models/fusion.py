"""Modality encoders + cross-modal fusion transformer (B1 unimodal, B4 fusion)."""
import torch
import torch.nn as nn


class IMUEnc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, d, 5, padding=2), nn.GELU(), nn.BatchNorm1d(d),
            nn.Conv1d(d, d, 5, padding=2, stride=2), nn.GELU(), nn.BatchNorm1d(d),
            nn.Conv1d(d, d, 5, padding=2, stride=2), nn.GELU(), nn.BatchNorm1d(d))

    def forward(self, x):                        # (B, 100, 6)
        return self.net(x.transpose(1, 2)).transpose(1, 2)   # (B, 25, d)


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


ENCODERS = dict(video=VideoEnc, skel=SkelEnc, imu=IMUEnc)
TOKENS = dict(video=4, skel=40, imu=25)


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

    def forward(self, batch, return_emb=False):
        B = batch[self.mods[0]].shape[0]
        tokens, emb = [self.cls.expand(B, -1, -1)], {}
        for m in self.mods:
            t = self.enc[m](batch[m]) + self.type_emb[m] + self.pos_emb[m]
            emb[m] = t.mean(dim=1)               # per-modality embedding (for the MI module later)
            tokens.append(t)
        h = self.norm(self.fusion(torch.cat(tokens, dim=1)))
        logits = self.head(h[:, 0])
        return (logits, emb) if return_emb else logits
