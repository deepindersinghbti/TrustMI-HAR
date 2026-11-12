"""Tests for the IMU-off bench (docs/plan_followup_2026-11-12.md, Part A, check A.7 (4)).
Synthetic tests need nothing; the last test needs the built WearGait-PD data and is skipped without it.
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_imuoff.py -q
"""
import math, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import cert_conditions as CC                                                    # noqa: E402

T = 20


def fake_x(n_imus, N=50, seed=0):
    """Non-zero normalised data, so a 0 can only come from switching off."""
    r = np.random.default_rng(seed)
    x = r.normal(size=(N, T, 6 * n_imus)).astype(np.float32)
    x[x == 0] = 0.5
    return x


def off_imus(x0, x1):
    """Per clip, the set of IMUs whose 6 channels are all exactly 0 over the whole clip in x1."""
    n = x0.shape[2] // 6
    z = (x1.reshape(len(x1), T, n, 6) == 0).all(axis=(1, 3))
    return z


@pytest.mark.parametrize("group", list(CC.IMUS))
def test_exactly_k_imus_off_rest_untouched(group):
    n = CC.IMUS[group]
    x = fake_x(n)
    gidx = np.arange(len(x)) * 7 + 3
    for k in range(1, n + 1):
        y = CC.imuoff_input(x, group, gidx, k)
        assert y.dtype == np.float32 and y.shape == x.shape
        z = off_imus(x, y)
        assert (z.sum(1) == k).all()                                   # exactly k IMUs per clip
        ch = np.repeat(z, 6, axis=1)[:, None, :].repeat(T, 1)
        assert (y[ch] == 0).all() and not np.signbit(y[ch]).any()      # +0.0, the training mean
        assert np.array_equal(y[~ch], x[~ch])                          # everything else byte-identical
    assert np.count_nonzero(CC.imuoff_input(x, group, gidx, n)) == 0   # k = all: the whole group is r_fill


@pytest.mark.parametrize("group", list(CC.IMUS))
def test_nested_and_deterministic(group):
    n = CC.IMUS[group]
    gidx = np.arange(40) + 100
    prev = np.zeros((40, n), bool)
    for k in range(1, n + 1):
        a = CC.imus_off(group, gidx, k)
        assert np.array_equal(a, CC.imus_off(group, gidx, k))          # same on a second call
        assert not (prev & ~a).any()                                   # k contains k - 1 (nested)
        assert a.sum() == 40 * k
        prev = a


def test_order_depends_on_clip_not_on_row_position():
    gidx = np.arange(60) + 5000
    perm = np.random.default_rng(1).permutation(60)
    a = CC.imus_off("legs", gidx, 3)
    b = CC.imus_off("legs", gidx[perm], 3)
    assert np.array_equal(a[perm], b)
    # and on the group: arms / trunk / legs orders are drawn separately
    assert not np.array_equal(CC.imus_off("trunk", gidx, 1), CC.imus_off("trunk", gidx + 1, 1))


def test_every_imu_off_equally_often():
    gidx = np.arange(8000)
    for group, n in CC.IMUS.items():
        first = CC.imus_off(group, gidx, 1).mean(0)                   # share of clips where IMU j goes first
        assert np.allclose(first, 1 / n, atol=0.025), (group, first)
        assert len({tuple(CC.imu_order(group, c)) for c in range(200)}) > min(20, math.factorial(n) // 2)


class FakeDS:
    def __init__(self, N=30):
        self.x = {g: fake_x(n, N, i) for i, (g, n) in enumerate(CC.IMUS.items())}
        self.y = np.zeros(N, int); self.idx = np.arange(N) + 11; self.subject = np.full(N, "P1")


def test_conditions_format():
    ds = FakeDS()
    conds, gidx = CC.build_imuoff_conditions("d3", ds, None, 0)
    want = ["clean"] + [f"io_fill:{g}:k{k}" for g, n in CC.IMUS.items() for k in range(1, n + 1)]
    assert list(conds) == want and len(want) == 14
    assert np.array_equal(gidx, ds.idx)
    for name, (repl, donor_y, avail, att) in conds.items():
        assert avail.all() and (donor_y == -1).all()
        if name == "clean":
            assert repl == {} and att == ""
        else:
            g = name.split(":")[1]
            assert att == g and list(repl) == [g] and repl[g].shape == ds.x[g].shape
    assert CC.parts_for("imuoff") == ("test",) and CC.conditions_suffix("imuoff") == "_imuoff"
    assert CC.folds_for("imuoff", "d3", [0, 1]) == [0, 1]
    with pytest.raises(SystemExit):
        CC.folds_for("imuoff", "d2v", [0])
    with pytest.raises(ValueError):
        CC.build_imuoff_conditions("d2v", ds, None, 0)


def test_other_condition_sets_unchanged():
    assert CC.parts_for("lies") == ("val", "test") and CC.parts_for("realdrop") == ("test", "dropout")
    assert CC.conditions_suffix("lies") == "" and CC.conditions_suffix("realdrop") == "_realdrop"


def test_all_off_equals_r_fill_on_real_data():
    """k = all IMUs of a group must equal r_fill of that group (check A.7 (3), here on the inputs)."""
    from trustmi.data import weargait as W
    if not W.DATA.exists():
        pytest.skip("run scripts/build_weargait.py first")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tr, te = W.WearGait(0, "train", stats_dir=tmp), W.WearGait(0, "test", stats_dir=tmp)
        rd, g1 = CC.build_realdrop_conditions("d3", te, tr, 0)
    io, g2 = CC.build_imuoff_conditions("d3", te, tr, 0)
    assert np.array_equal(g1, g2)
    for g, n in CC.IMUS.items():
        a, b = rd[f"r_fill:{g}"][0][g], io[f"io_fill:{g}:k{n}"][0][g]
        assert a.dtype == b.dtype and np.array_equal(a, b)
        one = io[f"io_fill:{g}:k1"][0][g]
        x = te.x[g].numpy()
        assert (np.count_nonzero(one != x, axis=(1, 2)) <= 6 * x.shape[1]).all()   # at most one IMU changes
