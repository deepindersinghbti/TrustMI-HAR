"""Tests for the no-watch loader (docs/plan_nowatch_2026-11-11.md, step 2). Run on the real data:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_nowatch.py -q
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trustmi.data import mmact as M                                              # noqa: E402

FOLDS = M.nowatch_folds()


@pytest.fixture(scope="module")
def npz():
    with np.load(M.TENSORS) as d:
        return {k: d[k] for k in ("status", "subject", "trial", "y", "scene")}


@pytest.fixture(scope="module", params=[1, 10])          # fold 1 = person 2 (126 clips), fold 10 = person 11 (6)
def built(request, tmp_path_factory):
    fold = request.param
    tmp = tmp_path_factory.mktemp(f"stats{fold}")
    nw = M.MMActNoWatch(fold)
    te = M.MMAct(fold, "test", stats_dir=tmp, video=True)
    tr = M.MMAct(fold, "train", stats_dir=tmp, video=True)
    return fold, nw, te, tr


# ---------------------------------------------------------------- which clips
def test_clip_counts_and_folds(npz):
    rows = M.nowatch_rows()
    assert len(rows) == M.N_NOWATCH == 247
    assert FOLDS == [1, 9, 10, 12, 15, 19]                        # persons 2, 10, 11, 13, 16, 20
    per_fold = [M.nowatch_rows(f) for f in range(20)]
    allr = np.concatenate(per_fold)
    assert len(allr) == 247 and len(np.unique(allr)) == 247       # each clip in exactly one fold
    assert set(allr) == set(rows)
    assert (npz["status"][rows] == "no_watch").all()              # never a main clip
    sizes = {f: len(r) for f, r in enumerate(per_fold) if len(r)}
    assert sizes == {1: 126, 9: 5, 10: 6, 12: 28, 15: 54, 19: 28}


def test_clip_ids_never_clash_with_main(built):
    fold, nw, te, _ = built
    assert nw.idx.min() >= M.NOWATCH_CLIP0 > M.N_MAIN3
    assert not np.isin(nw.idx, te.idx).any()


# ---------------------------------------------------------------- how the clips are built
def test_window_rows_reproduces_main_windows():
    """_window_rows on the main rows must give exactly what _windows() gives (same crop, padding, channels)."""
    x_main, lens_main, meta, _ = M._windows()
    with np.load(M.TENSORS) as d:
        rows = np.flatnonzero(d["status"] == "main")
        x, lens = M._window_rows(d, rows)
        assert (d["trial"][rows] == meta["trial"]).all()
    assert np.array_equal(lens, lens_main)
    for k in ("watch", "phone"):
        assert np.array_equal(x[k], x_main[k]), k


def test_watch_is_exactly_zero_and_no_nan(built):
    _, nw, _, _ = built
    assert torch.count_nonzero(nw.x["watch"]) == 0
    for k in ("watch", "phone", "video"):
        assert torch.isfinite(nw.x[k]).all(), k                   # the stored watch columns are NaN
    assert nw.x["watch"].shape[1:] == (M.T, 3) and nw.x["phone"].shape[1:] == (M.T, 10)
    assert nw.x["video"].shape[1:] == (4, 512)


def test_padding_is_zero(built):
    _, nw, _, _ = built
    L = nw.x["len"].numpy()
    for i in range(len(L)):
        assert torch.count_nonzero(nw.x["phone"][i, L[i]:]) == 0
        assert torch.count_nonzero(nw.x["phone"][i, :L[i]]) > 0


def test_same_statistics_as_the_main_loader(built):
    """Normalisation statistics must equal those MMAct(fold, ..., video=True) writes for the same fold."""
    _, nw, te, _ = built
    with np.load(te.stats_file) as f:
        for k, (mu, sd) in nw.stats.items():
            assert np.array_equal(f[f"{k}_mean"], mu), k
            assert np.array_equal(f[f"{k}_std"], sd), k


def test_same_normalisation_as_the_main_loader(built):
    """Apply the no-watch code path to the test person's MAIN clips: must equal MMAct's test inputs exactly."""
    _, nw, te, _ = built
    x_all, lens_all, meta, _ = M._with_video()
    rows = te.idx
    valid = np.arange(M.T)[None, :, None] < lens_all[rows, None, None]
    mu, sd = nw.stats["phone"]
    phone = np.where(valid, (x_all["phone"][rows] - mu) / sd, 0).astype(np.float32)
    assert np.array_equal(phone, te.x["phone"].numpy())
    vmu, vsd = nw.stats["video"]
    video = ((x_all["video"][rows].astype(np.float32) - vmu) / vsd).astype(np.float32)
    assert np.array_equal(video, te.x["video"].numpy())


def test_video_matched_by_name(built):
    _, nw, _, _ = built
    with np.load(M.VIDEO) as f:
        trial, status3, tokens = f["trial"], f["status3"], f["tokens"]
    pos = {t: i for i, t in enumerate(trial)}
    r = [pos[t] for t in nw.trial]
    assert (status3[r] == M.NOWATCH_STATUS3).all()
    vmu, vsd = nw.stats["video"]
    expect = ((tokens[r].astype(np.float32) - vmu) / vsd).astype(np.float32)
    assert np.array_equal(expect, nw.x["video"].numpy())


def test_labels_and_metadata(built, npz):
    fold, nw, te, _ = built
    rows = nw.idx - M.NOWATCH_CLIP0
    assert np.array_equal(nw.y.numpy(), npz["y"][rows])
    assert np.array_equal(nw.trial, npz["trial"][rows])
    assert np.array_equal(nw.group, npz["scene"][rows])
    assert (nw.subject == te.subject[0]).all()                    # the fold's test person
    assert np.array_equal(nw.classes, te.classes)


def test_watch_fill_is_the_r_fill_rule(built):
    """The silent-filled watch must be numerically what the r_fill operator gives (zeros, same shape)."""
    fold, nw, te, tr = built
    from trustmi.operators import OperatorBank
    bank = OperatorBank("d2v", "watch", te, tr, fold)
    xn, absent, avail, _ = bank.apply("r_fill")
    assert avail.all() and not absent.any()
    assert xn.shape[1:] == tuple(nw.x["watch"].shape[1:])
    assert np.count_nonzero(xn) == 0
