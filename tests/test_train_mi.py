"""Plan step 3: the matcher training script on FAKE data (no dataset files, CPU).
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s tests/test_train_mi.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import train_mi as T
from trustmi.mi import MIMatcher, sensor_encoder


def fake_mmact(n=48, seed=0, T_=750):
    """Two sensors that share a hidden 'action' signal; variable valid lengths; 2 subjects x 2 scenes."""
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(200, T_ + 1, (n,), generator=g)
    act = torch.randint(0, 6, (n,), generator=g)
    t = torch.arange(T_)[None, :, None].float()
    base = torch.sin(t * (0.02 + 0.01 * act[:, None, None]))
    watch = base.repeat(1, 1, 3) + 0.3 * torch.randn(n, T_, 3, generator=g)
    phone = base.repeat(1, 1, 10) + 0.3 * torch.randn(n, T_, 10, generator=g)
    valid = (torch.arange(T_)[None, :] < lens[:, None])[..., None]
    x = dict(watch=watch * valid, phone=phone * valid, len=lens)
    return SimpleNamespace(x=x, y=act, subject=np.repeat([1, 2], n // 2), group=np.tile([0, 1], n // 2))


def test_candidates_same_subject_and_group_only():
    ds = fake_mmact()
    c = T.Candidates(ds, K=4)
    C, V = c.draw(np.arange(len(ds.y)), np.random.default_rng(0))
    assert (C[:, 0] == np.arange(len(ds.y))).all()
    for r in range(len(C)):
        for j in C[r, 1:][V[r, 1:]]:
            assert j != r and ds.subject[j] == ds.subject[r] and ds.group[j] == ds.group[r]


def test_wrong_partners_are_stretched_to_the_anchor_length():
    ds = fake_mmact()
    c = T.Candidates(ds, K=4)
    rng = np.random.default_rng(1)
    C, V = c.draw(np.arange(8), rng)
    L = c.stretch_lengths(C, rng, stretch_pos=False)
    b = T.gather(ds.x, C, L, ["watch", "phone"])
    lens = ds.x["len"].numpy()
    out_len = b["len"].numpy().reshape(C.shape)
    assert (out_len == lens[C[:, :1]]).all()                       # every candidate has the anchor's length
    w = b["watch"].reshape(*C.shape, 750, 3)
    for r in range(8):
        for k in range(C.shape[1]):
            assert (w[r, k, lens[r]:] == 0).all()                  # nothing beyond the anchor's length
    assert torch.equal(w[:, 0], ds.x["watch"][torch.as_tensor(C[:, 0])])   # true partner untouched


def test_true_partner_detour_keeps_length():
    ds = fake_mmact()
    c = T.Candidates(ds, K=4)
    rng = np.random.default_rng(2)
    C, V = c.draw(np.arange(16), rng)
    src, dst, detour = c.stretch_lengths(C, rng, stretch_pos=True)
    assert (detour >= 0).any() and (detour < 0).any()
    b = T.gather(ds.x, C, (src, dst, detour), ["watch", "phone"])
    w = b["watch"].reshape(*C.shape, 750, 3)[:, 0]
    orig = ds.x["watch"][torch.as_tensor(C[:, 0])]
    lens = ds.x["len"].numpy()[C[:, 0]]
    for r in range(16):
        assert (w[r, lens[r]:] == 0).all()
        changed = not torch.equal(w[r], orig[r])
        assert changed == (detour[r] >= 0 and detour[r] != lens[r])


def test_training_loop_learns_on_fake_data():
    torch.manual_seed(0)
    tr, va = fake_mmact(64, 0), fake_mmact(32, 1)
    mods = ["watch", "phone"]
    model = MIMatcher({m: sensor_encoder(m, d=32, layers=1) for m in mods}, d=32, d_out=16)
    args = SimpleNamespace(lr=1e-3, wd=0.0, epochs=4, patience=10, bs=16, seed=0, fold=0)
    state, ep, log = T.train(model, tr.x, T.Candidates(tr, 4), va.x, T.Candidates(va, 4), mods, args, "cpu",
                             log_every=1)
    assert state is not None and 0 <= ep < 4
    assert log[-1]["train_loss"] < log[0]["train_loss"]
    assert all(np.isfinite(r["val_loss"]) for r in log)
