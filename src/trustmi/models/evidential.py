"""Literature baselines, re-implemented from the papers (the official repositories carry no licence):
  TMC   Han et al., "Trusted Multi-View Classification", ICLR 2021   -- Dempster's rule, pairwise in order
  ECML  Xu et al., "Reliable Conflictive Multi-View Learning", AAAI 2024 -- sequential pairwise evidence averaging
                                                                          + conflictive-degree consistency loss
Numerical equivalence with the official functions is checked by scripts/check_evidential_equivalence.py.

Each sensor ("view") gets the same network as our single-sensor models (FusionTransformer with one modality);
its output passes through Softplus to give non-negative evidence e, with Dirichlet alpha = e + 1,
uncertainty u = K / sum(alpha), projected probability p = alpha / sum(alpha).
forward() returns log p of the fused opinion, so train.py's accuracy / macro-F1 / msp code works unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from trustmi.models.fusion import FusionTransformer


# ------------------------------------------------------------------ evidential loss (both papers)
def kl_to_uniform(alpha):
    """KL( Dir(alpha) || Dir(1) ), per sample."""
    K = alpha.shape[1]
    s = alpha.sum(1, keepdim=True)
    first = (torch.lgamma(s) - torch.lgamma(alpha).sum(1, keepdim=True)
             - torch.lgamma(torch.tensor(float(K), device=alpha.device)))
    second = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(s))).sum(1, keepdim=True)
    return first + second


def edl_digamma(alpha, y, coef):
    """Per-sample evidential cross-entropy (digamma form) + annealed KL on the non-target evidence."""
    y1 = F.one_hot(y, alpha.shape[1]).to(alpha.dtype)
    s = alpha.sum(1, keepdim=True)
    a = (y1 * (torch.digamma(s) - torch.digamma(alpha))).sum(1, keepdim=True)
    return a + coef * kl_to_uniform((alpha - 1) * (1 - y1) + 1)


# ------------------------------------------------------------------ fusion rules
def fuse_tmc(alphas):
    """Dempster's combination of Dirichlet opinions, pairwise in view order (TMC)."""
    K = alphas[0].shape[1]

    def two(a0, a1):
        s0, s1 = a0.sum(1, keepdim=True), a1.sum(1, keepdim=True)
        b0, b1, u0, u1 = (a0 - 1) / s0, (a1 - 1) / s1, K / s0, K / s1
        bb = torch.bmm(b0.unsqueeze(2), b1.unsqueeze(1))                         # (B, K, K)
        C = bb.sum((1, 2)) - torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)        # conflict mass
        b = (b0 * b1 + b0 * u1 + b1 * u0) / (1 - C).unsqueeze(1)
        u = u0 * u1 / (1 - C).unsqueeze(1)
        return b * (K / u) + 1

    out = alphas[0]
    for a in alphas[1:]:
        out = two(out, a)
    return out


def fuse_ecml(evidences):
    """Sequential pairwise evidence averaging, in view order (ECML's released model)."""
    out = evidences[0]
    for e in evidences[1:]:
        out = (e + out) / 2
    return out + 1                                                               # -> alpha


def conflict_degree_loss(evidences):
    """ECML's consistency term: mean over the batch of the summed pairwise conflictive degrees."""
    V, K = len(evidences), evidences[0].shape[1]
    alphas = [e + 1 for e in evidences]
    p = torch.stack([a / a.sum(1, keepdim=True) for a in alphas])               # (V, B, K)
    u = torch.stack([K / a.sum(1) for a in alphas])                              # (V, B)
    total = 0
    for i in range(V):
        pd = (p - p[i]).abs().sum(2) / 2 / (V - 1)                               # projected distance
        cc = (1 - u[i]) * (1 - u)                                                # conjunctive certainty
        total = total + (pd * cc).sum(0)
    return total.mean()


def method_loss(method, evidences, y, coef, gamma=1.0):
    """TMC: sum of the per-view and fused losses. ECML: their average, plus gamma x conflictive degree."""
    alphas = [e + 1 for e in evidences]
    a_f = fuse_tmc(alphas) if method == "tmc" else fuse_ecml(evidences)
    terms = [edl_digamma(a_f, y, coef).mean()] + [edl_digamma(a, y, coef).mean() for a in alphas]
    if method == "tmc":
        return sum(terms)
    return sum(terms) / len(terms) + gamma * conflict_degree_loss(evidences)


# ------------------------------------------------------------------ the model
class EvidentialLateFusion(nn.Module):
    def __init__(self, modalities, method, n_classes, d=128, layers=2, anneal_epochs=6, gamma=1.0):
        super().__init__()
        assert method in ("tmc", "ecml")
        self.mods, self.method, self.K = list(modalities), method, n_classes
        self.anneal_epochs, self.gamma = anneal_epochs, gamma
        self.views = nn.ModuleDict({m: FusionTransformer([m], n_classes=n_classes, d=d, layers=layers)
                                    for m in self.mods})

    def evidences(self, batch):
        return [F.softplus(self.views[m](batch).float()) for m in self.mods]    # float32 from here on

    def fuse(self, evidences):
        return fuse_tmc([e + 1 for e in evidences]) if self.method == "tmc" else fuse_ecml(evidences)

    def forward(self, batch):
        a = self.fuse(self.evidences(batch))
        return torch.log(a / a.sum(1, keepdim=True))                             # log projected probability

    def loss(self, batch, y, epoch):
        """epoch is 0-based here; both papers count from 1, so the KL annealing uses epoch + 1."""
        coef = min(1.0, (epoch + 1) / self.anneal_epochs)
        return method_loss(self.method, self.evidences(batch), y, coef, self.gamma)
