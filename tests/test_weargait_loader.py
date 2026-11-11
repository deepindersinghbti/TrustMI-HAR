"""Tests for the WearGait-PD folds and loader, on the built data (run after build_weargait.py and
make_splits_weargait.py):
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_weargait_loader.py -q
"""
import json, sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trustmi.data import weargait as W                                          # noqa: E402

pytestmark = pytest.mark.skipif(not W.DATA.exists(), reason="run scripts/build_weargait.py first")
FOLDS = (0, 9)


@pytest.fixture(scope="module", params=FOLDS)
def parts(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(f"stats{request.param}")
    return request.param, {p: W.WearGait(request.param, p, stats_dir=tmp) for p in W.PARTS}


def test_folds_are_person_wise_and_balanced():
    split = json.loads(W.SPLITS.read_text())
    arrays, C, _ = W._load()
    cohort = C.groupby("person").group.first()
    seen = []
    for f in split["folds"]:
        t, v, r = set(f["test"]), set(f["val"]), set(f["train"])
        assert not (t & v) and not (t & r) and not (v & r)
        assert t | v | r == set(cohort.index)
        assert f["val"] == split["folds"][(f["fold"] + 1) % len(split["folds"])]["test"]
        n_pd = (cohort[f["test"]] == "PD").sum()
        assert n_pd == 10 and len(f["test"]) - n_pd in (8, 9)        # plan 2.5 (100 PD, 85 controls)
        seen += f["test"]
    assert sorted(seen) == sorted(cohort.index)                       # every person tested exactly once


def test_parts_hold_the_right_clips(parts):
    fold, P = parts
    arrays, C, split = W._load()
    f = split["folds"][fold]
    for p in ("train", "val", "test"):
        assert set(P[p].subject) <= set(f[p])
        assert C.clean.to_numpy()[P[p].idx].all()
    assert set(P["dropout"].subject) <= set(f["test"])
    assert not C.clean.to_numpy()[P["dropout"].idx].any()
    assert (P["dropout"].missing != "").all() and (P["test"].missing == "").all()
    all_test = np.flatnonzero(np.isin(arrays["person"], f["test"]))
    assert sorted(np.r_[P["test"].idx, P["dropout"].idx]) == sorted(all_test)   # nothing lost, nothing twice


def test_train_only_statistics(parts):
    fold, P = parts
    arrays, C, split = W._load()
    rows = np.flatnonzero(C.clean.to_numpy() & np.isin(arrays["person"], split["folds"][fold]["train"]))
    for s in W.SENSORS:
        mu = arrays[s][rows].astype(np.float64).mean((0, 1))
        assert np.allclose(P["test"].stats[s][0], mu, atol=1e-6)
        assert np.array_equal(P["dropout"].stats[s][0], P["train"].stats[s][0])
    z = P["train"].x["legs"].numpy()
    assert np.abs(z.mean((0, 1))).max() < 1e-3 and np.abs(z.std((0, 1)) - 1).max() < 1e-2
    with np.load(P["train"].stats_file) as f:
        assert np.array_equal(f["arms_mean"], P["train"].stats["arms"][0])


def test_inputs_are_finite_and_missing_is_training_mean(parts):
    _, P = parts
    arrays, _, _ = W._load()
    for p, ds in P.items():
        for s in W.SENSORS:
            assert torch.isfinite(ds.x[s]).all(), (p, s)
        assert (ds.x["len"] == W.T).all()
        assert ds.x["arms"].shape[1:] == (100, 12) and ds.x["trunk"].shape[1:] == (100, 18)
        assert ds.x["legs"].shape[1:] == (100, 48)
    d = P["dropout"]
    if len(d):
        for s in W.SENSORS:
            nan = np.isnan(arrays[s][d.idx])
            assert (d.x[s].numpy()[nan] == 0).all()                    # missing -> 0 = training mean


def test_labels_and_metadata(parts):
    _, P = parts
    arrays, C, _ = W._load()
    for ds in P.values():
        assert np.array_equal(ds.y.numpy(), arrays["y"][ds.idx])
        assert set(ds.group) <= {"PD", "control"}
        assert list(ds.classes) == ["Walk", "Standing", "TandemWalk", "Turn", "Sitting", "TurnToSit"]
        assert len(set(ds.trial)) == len(ds.trial)
    assert set(P["train"].y.tolist()) == set(range(6))
