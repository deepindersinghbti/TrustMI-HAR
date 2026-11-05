"""Journal (D): loaders with video as a 3rd MMAct sensor (d2v) and whole-frame UTD-MHAD video (d1v2).
Real data, CPU only. Tests that need a feature file skip until it exists.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s tests/test_video_loaders.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trustmi.data import mmact as M
from trustmi.data import utd as U
from trustmi.models.fusion import FusionTransformer
from trustmi.models.evidential import EvidentialLateFusion

need_mm = pytest.mark.skipif(not M.VIDEO.exists(), reason="MMAct video features not merged yet")
need_utd = pytest.mark.skipif(not U.VIDEO["v2"].exists(), reason="UTD-MHAD v2 video features not extracted yet")


@need_mm
def test_d2v_rows_match_by_name():
    x3, lens3, meta3, _ = M._with_video()
    x2, lens2, meta2, _ = M._windows()
    with np.load(M.VIDEO) as f:
        v = {k: f[k] for k in ("trial", "status3", "tokens")}
    assert len(meta3["trial"]) == M.N_MAIN3 == 8432
    pos2 = {t: i for i, t in enumerate(meta2["trial"])}
    posv = {t: i for i, t in enumerate(v["trial"])}
    for r in np.random.default_rng(0).choice(len(meta3["trial"]), 200, replace=False):
        t = meta3["trial"][r]
        assert v["status3"][posv[t]] == "main3"
        assert np.array_equal(x3["video"][r], v["tokens"][posv[t]].astype(np.float32))      # right video
        i = pos2[t]
        assert np.array_equal(x3["watch"][r], x2["watch"][i]) and np.array_equal(x3["phone"][r], x2["phone"][i])
        assert lens3[r] == lens2[i] and meta3["y"][r] == meta2["y"][i] and meta3["subject"][r] == meta2["subject"][i]


@need_mm
def test_d2v_folds_disjoint_and_each_clip_tested_once():
    seen = np.zeros(M.N_MAIN3, int)
    for fold in range(20):
        tr, va, te = (M.MMAct(fold, p, video=True) for p in ("train", "val", "test"))
        s = [set(d.subject.tolist()) for d in (tr, va, te)]
        assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
        assert len(tr) + len(va) + len(te) == M.N_MAIN3
        assert set(tr.x) == {"watch", "phone", "video", "len"}
        seen[te.idx] += 1
    assert (seen == 1).all()


@need_mm
def test_d2v_video_standardised_on_train_only():
    tr = M.MMAct(0, "train", video=True)
    v = tr.x["video"].double()
    assert v.mean(dim=(0, 1)).abs().max() < 1e-4 and (v.std(dim=(0, 1)) - 1).abs().max() < 1e-2
    x, _, meta, split = M._with_video()
    f = split["folds"][0]
    tr_rows = np.flatnonzero(np.isin(meta["subject"], f["train"]))
    te_rows = np.flatnonzero(np.isin(meta["subject"], f["test"]))
    clean = x["video"][tr_rows].mean((0, 1))
    leaky = x["video"][np.concatenate([tr_rows, te_rows])].mean((0, 1))
    assert not np.allclose(clean, leaky), "statistics ignore the training rows?"
    assert (ROOT / "data/processed/norm_stats/mmact_v1_videov2_fold0.npz").exists()


def test_d2_unchanged():
    """The conference loader (no video) still gives 8,471 main clips with watch and phone only."""
    _, _, meta, _ = M._windows()
    assert len(meta["trial"]) == 8471
    te = M.MMAct(0, "test")
    assert set(te.x) == {"watch", "phone", "len"} and not te.video


@need_utd
def test_d1v2_same_trials_new_features():
    d1, _ = U._load_all("v1")
    d2, _ = U._load_all("v2")
    assert (d1["trial"] == d2["trial"]).all() and d2["video"].shape == (861, 4, 512)
    assert np.isfinite(d2["video"]).all() and not np.allclose(d1["video"], d2["video"])
    tr = U.UTDMHAD(0, "train", video="v2")
    v = tr.x["video"].double()
    assert v.mean(dim=(0, 1)).abs().max() < 1e-4 and (v.std(dim=(0, 1)) - 1).abs().max() < 1e-2
    assert (ROOT / "data/processed/norm_stats/utd_v1_videov2_fold0.npz").exists()


@need_mm
@pytest.mark.parametrize("mods", [["video"], ["watch", "phone", "video"], ["phone", "video"], ["watch", "video"]])
def test_models_accept_d2v_batch(mods):
    te = M.MMAct(0, "test", video=True)
    batch = {k: v[:8] for k, v in te.x.items()}
    torch.manual_seed(0)
    for model in (FusionTransformer(mods, n_classes=35), EvidentialLateFusion(mods, "tmc", n_classes=35)):
        out = model.eval()(batch)
        assert out.shape == (8, 35) and torch.isfinite(out).all()
