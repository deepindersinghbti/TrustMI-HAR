"""Tests for the 14 Nov additions (docs/plan_additions_2026-11-14.md, Checks 3): raw-zero watch (Addition 3) and
partial IMU-off (Addition 2). Synthetic tests need nothing; the data tests need the built data and are skipped
without it.
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_additions.py -q
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import cert_conditions as CC                                                    # noqa: E402

T = 40


# ---------------------------------------------------------------- Addition 3: raw zeros
def test_raw_zero_values_and_padding():
    r = np.random.default_rng(0)
    N, C = 30, 3
    x = r.normal(size=(N, T, C)).astype(np.float32)
    lens = r.integers(5, T + 1, N)
    mu, sd = np.array([0.3, -9.7, 1.2]), np.array([2.0, 3.0, 0.5])
    z = CC.raw_zero(x, lens, mu, sd)
    assert z.dtype == np.float32 and z.shape == x.shape
    valid = np.arange(T)[None, :] < lens[:, None]
    for c in range(C):
        assert np.allclose(z[..., c][valid], (0 - mu[c]) / sd[c])          # raw 0, then the training normalisation
    assert (z[~valid] == 0).all()                                         # padding stays exactly 0
    back = z * sd + mu                                                    # undoing the normalisation gives raw 0
    assert np.allclose(back[valid], 0, atol=1e-5)
    assert np.array_equal(CC.raw_zero(x, lens, mu, sd), z)                 # deterministic
    assert np.array_equal(CC.raw_zero(np.zeros_like(x), lens, mu, sd), z)  # does not depend on the old values


def test_raw_zero_rejects_wrong_statistics():
    x = np.zeros((2, T, 3), np.float32)
    with pytest.raises(RuntimeError):
        CC.raw_zero(x, [T, T], np.zeros(4), np.ones(4))


def test_new_kinds_use_the_old_folds_and_parts():
    assert CC.parts_for("nowatch2") == CC.parts_for("nowatch") == ("test", "nowatch")
    assert CC.parts_for("imuoff2") == CC.parts_for("imuoff") == ("test",)
    assert CC.conditions_suffix("nowatch2") == "_nowatch2" and CC.conditions_suffix("imuoff2") == "_imuoff2"
    for k in ("lies", "faults", "nowatch", "realdrop", "imuoff"):        # the old ones are unchanged
        assert CC.base_kind(k) == k
    assert CC.conditions_suffix("lies") == "" and CC.conditions_suffix("faults") == "_faults"
    assert CC.conditions_suffix("faults", False) == "_faults-novideo"
    assert CC.parts_for("lies") == CC.parts_for("faults") == ("val", "test")


# ---------------------------------------------------------------- Addition 2: partial IMU-off
def fake_x(n_imus, N=60, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=(N, T, 6 * n_imus)).astype(np.float32)
    x[x == 0] = 0.5                                     # a 0 can only come from switching off
    return x


@pytest.mark.parametrize("group", list(CC.IMUS))
@pytest.mark.parametrize("p", CC.IMUOFF2_P)
def test_partial_exactly_k_imus_one_stretch(group, p):
    n = CC.IMUS[group]
    x = fake_x(n)
    gidx = np.arange(len(x)) * 7 + 3
    lens = np.full(len(x), T)
    for k in range(1, n + 1):
        y = CC.imuoff_part_input(x, group, gidx, k, p, lens)
        assert y.dtype == np.float32 and y.shape == x.shape
        off = CC.imus_off(group, gidx, k)                                  # the IMU-off test's own order
        for r in range(len(x)):
            a, m = CC.off_stretch(group, gidx[r], p, T)
            assert m == round(p / 100 * T) and 0 <= a and a + m <= T      # fully inside the clip
            ch = np.repeat(off[r], 6)
            mask = np.zeros((T, 6 * n), bool)
            mask[a: a + m, ch] = True
            assert (y[r][mask] == 0).all() and not np.signbit(y[r][mask]).any()
            assert np.array_equal(y[r][~mask], x[r][~mask])               # everything else byte-identical
            zero_t = np.flatnonzero((y[r][:, ch] == 0).all(axis=1))
            assert len(zero_t) == m and (np.diff(zero_t) == 1).all()       # one contiguous stretch


@pytest.mark.parametrize("group", list(CC.IMUS))
def test_partial_same_stretch_for_every_k_nested_and_deterministic(group):
    n = CC.IMUS[group]
    x = fake_x(n, N=25, seed=1)
    gidx = np.arange(25) + 500
    lens = np.full(25, T)
    for p in CC.IMUOFF2_P:
        prev = None
        for k in range(1, n + 1):
            y = CC.imuoff_part_input(x, group, gidx, k, p, lens)
            assert np.array_equal(y, CC.imuoff_part_input(x, group, gidx, k, p, lens))
            z = (y == 0)
            if prev is not None:
                assert not (prev & ~z).any()                               # k contains k - 1 (same stretch)
            prev = z
    starts = {p: [CC.off_stretch(group, c, p, T)[0] for c in gidx] for p in CC.IMUOFF2_P}
    assert len(set(starts[25])) > 3                                        # the position really varies by clip


def test_partial_differs_from_whole_clip_and_endpoint():
    x = fake_x(8, N=10)
    gidx, lens = np.arange(10), np.full(10, T)
    whole = CC.imuoff_input(x, "legs", gidx, 8)
    part = CC.imuoff_part_input(x, "legs", gidx, 8, 50, lens)
    assert np.count_nonzero(whole) == 0
    assert np.count_nonzero(part == 0) == 10 * 20 * 48                      # half the clip, all 48 channels


def test_build_imuoff2_conditions_names():
    class FakeDS:
        pass
    ds = FakeDS()
    ds.y = np.zeros(6, int); ds.idx = np.arange(6)
    ds.x = {g: fake_x(n, N=6) for g, n in CC.IMUS.items()}
    ds.x["len"] = np.full(6, T)
    conds, gidx = CC.build_imuoff2_conditions("d3", ds, None, 0)
    names = [c for c in conds if c != "clean"]
    assert len(names) == sum(CC.IMUS.values()) * len(CC.IMUOFF2_P) == 39
    assert all(conds[c][3] == c.split(":")[1] for c in names)              # attacked = the group
    with pytest.raises(ValueError):
        CC.build_imuoff2_conditions("d2v", ds, None, 0)


# ---------------------------------------------------------------- data tests (desktop only)
def _have_mmact():
    try:
        from trustmi.data import mmact
        return mmact.TENSORS.exists() and mmact.VIDEO.exists()
    except Exception:
        return False


@pytest.mark.skipif(not _have_mmact(), reason="MMAct data not built here")
def test_data_nowatch2_bench_fold19():
    from trustmi.data.mmact import MMAct, MMActNoWatch
    from trustmi.operators import _stats_for
    tr, te, nw = MMAct(19, "train", video=True), MMAct(19, "test", video=True), MMActNoWatch(19)
    mu, sd, _ = _stats_for("watch", 19, np.unique(tr.subject), te.stats_file)
    assert np.allclose(nw.stats["watch"][0], mu) and np.allclose(nw.stats["watch"][1], sd)   # same statistics
    for ds in (te, nw):
        old, _ = CC.build_nowatch_conditions("d2v", ds, tr, 19)
        new, _ = CC.build_nowatch2_conditions("d2v", ds, tr, 19)
        extra = set(new) - set(old)
        assert extra == ({"nw_zero:watch"} if ds is nw else {"r_zero:watch"})
        for c in old:                                                       # the old conditions, unchanged
            for a, b in zip(old[c][0].values(), new[c][0].values()):
                assert np.array_equal(a, b)
        xz = new[extra.pop()][0]["watch"]
        lens = np.asarray(ds.x["len"])
        valid = np.arange(xz.shape[1])[None, :] < lens[:, None]
        assert np.allclose((xz * sd + mu)[valid], 0, atol=1e-4)           # raw 0 after undoing the normalisation
        assert not np.isnan(xz).any()
