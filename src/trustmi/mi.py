"""Mutual information (MI) between sensors: InfoNCE matcher, dataset-level bound, per-clip pointwise MI.

Plan: docs/plan_mi_2026-11-04.md (section 3). In plain words:
  * every sensor m gets an encoder and a projection head; z_m = unit vector of 64 numbers
  * matching score f(z_m, z_n) = scale * <z_m, z_n>   (scale = exp(learned log-scale), as in CLIP)
  * InfoNCE: in a batch, each clip's sensor-m vector must pick out ITS OWN sensor-n vector among the
    candidates it is allowed to compare with ("negatives"). Negatives can be restricted to clips of the
    same subject (and same wrist/thigh group), exactly like the lie donors, via `groups`.
  * dataset-level MI lower bound:   I(X_m; X_n) >= E_i[ log K_i - CE_i ]
    (K_i = number of candidates of row i, positive included; the bound can never exceed log K_i)
  * per-clip pointwise MI against a fixed reference bank of K clips:
        i_mn(x) = f(z_m, z_n) - log( (1/K) sum_j exp f(z_m, z_n^(j)) ),  averaged over both directions
All MI values are in nats.
"""
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from trustmi.models.fusion import FusionTransformer

LOG_SCALE_INIT, LOG_SCALE_MAX = math.log(1 / 0.07), math.log(100.0)   # CLIP's start value and cap


class ProjHead(nn.Module):
    """Two-layer MLP, output normalised to unit length."""
    def __init__(self, d_in, d_out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def sensor_encoder(m, d=128, layers=2):
    """Same network as a witness for sensor m, but returning the CLS feature (d numbers) instead of class
    scores: the classification head is replaced by the identity. Used by the MI-own matcher."""
    enc = FusionTransformer([m], n_classes=1, d=d, layers=layers)
    enc.head = nn.Identity()
    return enc


class MIMatcher(nn.Module):
    """One encoder + projection head per sensor and one shared learned scale.

    encoders: {sensor: module taking the batch dict and returning (B, d) features}
      MI-own:    sensor_encoder(m) for every m (trained)
      MI-frozen: the witness models with head = Identity, frozen (only the heads and scale train)
    """
    def __init__(self, encoders, d=128, d_out=64, freeze_encoders=False):
        super().__init__()
        self.mods = list(encoders)
        self.enc = nn.ModuleDict(encoders)
        self.proj = nn.ModuleDict({m: ProjHead(d, d_out) for m in self.mods})
        self.log_scale = nn.Parameter(torch.tensor(LOG_SCALE_INIT))
        self.frozen = freeze_encoders
        if freeze_encoders:
            for p in self.enc.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):                  # frozen encoders stay in eval mode (BatchNorm, dropout)
        super().train(mode)
        if self.frozen:
            self.enc.eval()
        return self

    def scale(self):
        return self.log_scale.clamp(max=LOG_SCALE_MAX).exp()

    def embed(self, batch):
        out = {}
        for m in self.mods:
            sub = {m: batch[m]}
            if "len" in batch:
                sub["len"] = batch["len"]
            out[m] = self.proj[m](self.enc[m](sub).float())
        return out

    def forward(self, batch, groups=None):
        return info_nce(self.embed(batch), self.scale(), groups)


def candidate_mask(groups, n, device):
    """(n, n) bool: True where row i may be compared with column j (same group). None = everything."""
    if groups is None:
        return torch.ones(n, n, dtype=torch.bool, device=device)
    g = torch.as_tensor(groups, device=device)
    return g[:, None] == g[None, :]


def info_nce(z, scale, groups=None):
    """Symmetric InfoNCE over all sensor pairs.

    z: {sensor: (B, k) unit vectors of the SAME clips, row i = clip i};  scale: positive scalar tensor
    groups: optional (B,) ints; clips are only compared within their group (positive always included)
    Returns (loss, info) where info[(m, n)] = dict(ce=mean CE, mi_lb=mean(log K_i - CE_i)) over both
    directions; loss = mean CE over pairs and directions.
    """
    mods = list(z)
    B = z[mods[0]].shape[0]
    allowed = candidate_mask(groups, B, z[mods[0]].device)
    logK = allowed.sum(1).float().log()
    target = torch.arange(B, device=allowed.device)
    losses, info = [], {}
    for a, b in itertools.combinations(mods, 2):
        logits = scale * z[a] @ z[b].T                          # (B, B), row i: clip i's a vs every b
        logits = logits.masked_fill(~allowed, float("-inf"))
        ce_ab = F.cross_entropy(logits, target, reduction="none")
        ce_ba = F.cross_entropy(logits.T, target, reduction="none")   # allowed is symmetric
        ce = 0.5 * (ce_ab + ce_ba)
        losses.append(ce.mean())
        info[(a, b)] = dict(ce=ce.mean().detach(), mi_lb=(logK - ce).mean().detach())
    return torch.stack(losses).mean(), info


@torch.no_grad()
def pointwise_mi(za, zb, bank_a, bank_b, scale):
    """Per-clip pointwise MI between sensors a and b, symmetric (mean of both directions).

    za, zb:         (N, k) unit vectors of the N clips to judge (row i of za belongs with row i of zb)
    bank_a, bank_b: (K, k) unit vectors of K fixed reference clips (e.g. training clips)
    i(a->b) = f(za_i, zb_i) - logmeanexp_j f(za_i, bank_b_j);  i(b->a) likewise;  result = their mean.
    High = sensors a and b of this clip belong together; low = one of them looks foreign.
    """
    K = bank_a.shape[0]
    pos = scale * (za * zb).sum(1)
    norm_ab = torch.logsumexp(scale * za @ bank_b.T, 1) - math.log(K)
    norm_ba = torch.logsumexp(scale * zb @ bank_a.T, 1) - math.log(K)
    return 0.5 * ((pos - norm_ab) + (pos - norm_ba))


@torch.no_grad()
def pairwise_pmi(z, bank, scale):
    """All sensor pairs: {(a, b): (N,) pointwise MI}. z and bank: {sensor: (N, k)} and {sensor: (K, k)}."""
    return {(a, b): pointwise_mi(z[a], z[b], bank[a], bank[b], scale)
            for a, b in itertools.combinations(list(z), 2)}


def mi_signals(pmi, mods):
    """Certificate inputs from pairwise pointwise MI (numpy or torch, shape (N,) each).

    s4            = -min over pairs of i_mn      (high = some pair does not belong together)
    mi_share_<m>  = mean over n != m of i_mn     (how much sensor m shares with the others)
    Naming (M >= 3): the sensor with the LOWEST mi_share_<m>.
    """
    pairs = list(pmi)
    vals = torch.stack([torch.as_tensor(pmi[p]) for p in pairs])
    out = {"mi_min": vals.min(0).values, "s4": -vals.min(0).values, "mi_mean": vals.mean(0)}
    for m in mods:
        mine = [torch.as_tensor(pmi[p]) for p in pairs if m in p]
        out[f"mi_share_{m}"] = torch.stack(mine).mean(0)
    return out
