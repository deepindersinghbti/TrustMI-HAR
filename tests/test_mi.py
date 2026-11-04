"""Correctness checks for src/trustmi/mi.py (plan step 1). Run: python -m pytest -q tests/test_mi.py

Known-answer tests: for correlated Gaussians the true MI is -0.5 * d * log(1 - rho^2) nats; for independent
data it is 0. The InfoNCE estimate is a lower bound, so it must come out close to and not above the truth.
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustmi.mi import (MIMatcher, info_nce, pointwise_mi, pairwise_pmi, mi_signals,
                        sensor_encoder, candidate_mask)


class MLPEnc(nn.Module):
    """Toy encoder for the Gaussian tests: reads batch[name] of shape (B, d_in)."""
    def __init__(self, name, d_in, d=64):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, batch):
        return self.net(batch[self.name])


def gaussian_pairs(n, d, rho, gen):
    a = torch.randn(n, d, generator=gen)
    b = rho * a + math.sqrt(1 - rho ** 2) * torch.randn(n, d, generator=gen)
    return {"a": a, "b": b}


def fit_matcher(rho, d=4, steps=1500, bs=256, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MIMatcher({"a": MLPEnc("a", d), "b": MLPEnc("b", d)}, d=64, d_out=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        loss, _ = model(gaussian_pairs(bs, d, rho, gen))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    held = gaussian_pairs(bs, d, rho, torch.Generator().manual_seed(seed + 999))   # unseen data
    with torch.no_grad():
        _, info = model(held)
    return model, info[("a", "b")]["mi_lb"].item(), held


def test_gaussian_known_mi():
    d, rho = 4, 0.8
    true_mi = -0.5 * d * math.log(1 - rho ** 2)                   # 2.04 nats
    _, est, _ = fit_matcher(rho, d)
    print(f"Gaussian: true MI {true_mi:.3f} nats, InfoNCE estimate {est:.3f} (bound <= log 256 = 5.55)")
    assert est <= true_mi + 0.15, "a lower bound must not clearly exceed the truth"
    assert est >= 0.75 * true_mi, "estimate too far below the truth: estimator or training is broken"


def test_independent_is_zero():
    _, est, _ = fit_matcher(rho=0.0, d=4, steps=500)
    print(f"Independent: true MI 0, estimate {est:.3f}")
    assert abs(est) < 0.15


def test_pointwise_mi_separates_matched_from_mismatched():
    d, rho = 4, 0.8
    model, _, held = fit_matcher(rho, d)
    bank_data = gaussian_pairs(512, d, rho, torch.Generator().manual_seed(7))
    with torch.no_grad():
        z, bank = model.embed(held), model.embed(bank_data)
        pmi = pointwise_mi(z["a"], z["b"], bank["a"], bank["b"], model.scale())
        swapped = pointwise_mi(z["a"], z["b"].roll(1, 0), bank["a"], bank["b"], model.scale())   # "lie"
    print(f"Pointwise MI: matched mean {pmi.mean():.3f}, swapped mean {swapped.mean():.3f}")
    assert pmi.mean() > 1.0 and swapped.mean() < 0.0
    auroc = (pmi[:, None] > swapped[None, :]).float().mean().item()
    assert auroc > 0.9


def test_pointwise_mi_by_hand():
    za = torch.tensor([[1.0, 0.0]]); zb = torch.tensor([[1.0, 0.0]])
    bank_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]]); bank_b = bank_a.clone()
    s = torch.tensor(2.0)
    # a->b: pos = 2; scores to bank_b = [2, 0] -> logmeanexp = log((e^2 + 1) / 2). b->a is identical.
    expect = 2.0 - math.log((math.exp(2) + 1) / 2)
    got = pointwise_mi(za, zb, bank_a, bank_b, s).item()
    assert got == pytest.approx(expect, abs=1e-6)


def test_group_mask_restricts_negatives():
    g = torch.Generator().manual_seed(0)
    z = {"a": nn.functional.normalize(torch.randn(6, 8, generator=g), dim=-1)}
    z["b"] = z["a"].clone()
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    m = candidate_mask(groups, 6, "cpu")
    assert m.sum(1).tolist() == [2] * 6 and m.equal(m.T)
    # every clip its own group: one candidate, CE = 0, log K = 0 -> bound exactly 0
    _, info = info_nce(z, torch.tensor(10.0), torch.arange(6))
    assert info[("a", "b")]["ce"].item() == pytest.approx(0.0, abs=1e-6)
    assert info[("a", "b")]["mi_lb"].item() == pytest.approx(0.0, abs=1e-6)
    # with the mask, a confusable clip OUTSIDE the group cannot be chosen
    za = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]); zb = za.clone()
    _, free = info_nce({"a": za, "b": zb}, torch.tensor(10.0))
    _, masked = info_nce({"a": za, "b": zb}, torch.tensor(10.0), torch.tensor([0, 0, 1]))
    assert masked[("a", "b")]["ce"] < free[("a", "b")]["ce"]


def test_signals_and_naming():
    pmi = {("imu", "skel"): torch.tensor([3.0, 0.5]), ("imu", "video"): torch.tensor([3.0, 0.4]),
           ("skel", "video"): torch.tensor([3.0, 2.9])}               # clip 2: the IMU looks foreign
    s = mi_signals(pmi, ["imu", "skel", "video"])
    assert s["s4"].tolist() == pytest.approx([-3.0, -0.4])
    shares = torch.stack([s[f"mi_share_{m}"] for m in ["imu", "skel", "video"]])
    assert shares[:, 1].argmin().item() == 0                          # names the IMU


@pytest.mark.parametrize("m,shape,extra", [("imu", (5, 100, 6), {}), ("skel", (5, 40, 20, 3), {}),
                                           ("video", (5, 4, 512), {}),
                                           ("watch", (5, 750, 3), {"len": torch.tensor([750, 400, 10, 750, 1])}),
                                           ("phone", (5, 750, 10), {"len": torch.tensor([750] * 5)})])
def test_sensor_encoder_shapes(m, shape, extra):
    enc = sensor_encoder(m).eval()
    out = enc({m: torch.randn(*shape), **extra})
    assert out.shape == (5, 128) and torch.isfinite(out).all()


def test_frozen_encoders_do_not_change():
    torch.manual_seed(0)
    mods = ["imu", "skel"]
    model = MIMatcher({m: sensor_encoder(m) for m in mods}, freeze_encoders=True)
    before = {k: v.clone() for k, v in model.enc.state_dict().items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    assert not model.enc.training                                    # BatchNorm statistics stay fixed
    batch = {"imu": torch.randn(8, 100, 6), "skel": torch.randn(8, 40, 20, 3)}
    loss, _ = model(batch)
    opt.zero_grad(); loss.backward(); opt.step()
    after = model.enc.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert loss.requires_grad


def test_pairwise_pmi_keys():
    z = {m: nn.functional.normalize(torch.randn(4, 8), dim=-1) for m in ["imu", "skel", "video"]}
    out = pairwise_pmi(z, z, torch.tensor(5.0))
    assert set(out) == {("imu", "skel"), ("imu", "video"), ("skel", "video")}
    assert all(v.shape == (4,) for v in out.values())


# ---------------------------------------------------------------- step 3 additions
from trustmi.mi import info_nce_candidates, resample_to


def test_resample_matches_lie_donor_code():
    """Our GPU stretching must equal trustmi.operators._resample, which builds the lie donors."""
    import numpy as np
    from trustmi.operators import _resample
    rng = np.random.default_rng(0)
    T = 50
    x = torch.zeros(6, T, 3, dtype=torch.float64)
    src = torch.tensor([50, 30, 7, 1, 50, 2]); dst = torch.tensor([20, 45, 7, 9, 1, 50])
    for p in range(6):
        x[p, :src[p]] = torch.from_numpy(rng.normal(size=(int(src[p]), 3)))
    out = resample_to(x, src, dst)
    for p in range(6):
        ref = _resample(x[p, :src[p]].numpy(), int(dst[p]))
        assert np.allclose(out[p, :dst[p]].numpy(), ref, atol=1e-12)
        assert (out[p, dst[p]:] == 0).all()                       # padding stays exactly 0
    sk = torch.randn(4, 40, 20, 3, dtype=torch.float64)          # extra trailing dims are fine
    assert torch.equal(resample_to(sk, torch.full((4,), 40), torch.full((4,), 40)), sk)


def test_candidate_infonce_bounds():
    B, K1, k = 8, 5, 16
    base = nn.functional.normalize(torch.randn(B, k), dim=-1)
    other = nn.functional.normalize(torch.randn(B, K1 - 1, k), dim=-1)
    zc = {m: torch.cat([base[:, None], other], 1) for m in ("a", "b")}     # true partner identical
    _, info = info_nce_candidates(zc, torch.tensor(100.0))
    assert info[("a", "b")]["mi_lb"].item() == pytest.approx(math.log(K1), abs=1e-3)   # hits the ceiling
    valid = torch.ones(B, K1, dtype=torch.bool); valid[:, 3:] = False
    _, info = info_nce_candidates(zc, torch.tensor(100.0), valid)
    assert info[("a", "b")]["mi_lb"].item() == pytest.approx(math.log(3), abs=1e-3)    # fewer candidates
    zr = {m: nn.functional.normalize(torch.randn(B, K1, k), dim=-1) for m in ("a", "b")}
    _, info = info_nce_candidates(zr, torch.tensor(0.0))                  # uninformative critic
    assert info[("a", "b")]["mi_lb"].item() == pytest.approx(0.0, abs=1e-6)


def test_pointwise_mi_per_clip_bank_equals_shared_bank():
    za, zb = (nn.functional.normalize(torch.randn(5, 8), dim=-1) for _ in range(2))
    ba, bb = (nn.functional.normalize(torch.randn(7, 8), dim=-1) for _ in range(2))
    s = torch.tensor(3.0)
    shared = pointwise_mi(za, zb, ba, bb, s)
    per = pointwise_mi(za, zb, ba.expand(5, -1, -1), bb.expand(5, -1, -1), s)
    assert torch.allclose(shared, per, atol=1e-6)
