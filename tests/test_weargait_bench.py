"""Tests for the WearGait-PD benches: operators and faults on multi-IMU sensors, donor rule, real-dropout
conditions; plus a check that the operators of the other datasets are unchanged. Needs the built data.
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_weargait_bench.py -q
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from trustmi.data import weargait as W                                          # noqa: E402
from trustmi import operators as O                                               # noqa: E402
from trustmi import faults as F                                                  # noqa: E402

pytestmark = pytest.mark.skipif(not W.DATA.exists(), reason="run scripts/build_weargait.py first")
N_IMU = dict(arms=2, trunk=3, legs=8)


@pytest.fixture(scope="module")
def fold0(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stats")
    tr, te = W.WearGait(0, "train", stats_dir=tmp), W.WearGait(0, "test", stats_dir=tmp)
    return tr, te, W.WearGait(0, "dropout", stats_dir=tmp)


def phys(bank, x):
    return x.reshape(len(x), bank.T, bank.C) * bank.sd + bank.mu


def test_layouts_match_the_loader(fold0):
    _, te, _ = fold0
    for s, n in N_IMU.items():
        L = O.LAYOUTS[("d3", s)]
        assert len(L.imus) == n and len(L.accel) == len(L.gyro) == 3 * n
        assert max(L.accel + L.gyro) + 1 == te.x[s].shape[2] == 6 * n
        assert not L.ang_sin and not L.generic


def test_each_imu_rotated_on_its_own(fold0):
    """h_rotate: every IMU keeps its vector lengths (a rotation) and gets a different rotation."""
    tr, te, _ = fold0
    b = O.OperatorBank("d3", "legs", te, tr, 0)
    rows = np.arange(20)
    xn, _, avail, info = b.apply("h_rotate", rows=rows, level=3)
    p0, p1 = phys(b, b.x[rows]), phys(b, xn)
    clean = np.array([i["clipped"] == 0 for i in info])
    assert clean.sum() >= 5
    for r in np.flatnonzero(clean)[:5]:
        Rs = []
        for acc, gyr in O.LAYOUTS[("d3", "legs")].imus:
            for tri in (acc, gyr):
                a, c = p0[r][:, list(tri)], p1[r][:, list(tri)]
                assert np.allclose(np.linalg.norm(a, axis=1), np.linalg.norm(c, axis=1), atol=1e-6)
            a, c = p0[r][:, list(acc)], p1[r][:, list(acc)]
            Rs.append(np.linalg.lstsq(a, c, rcond=None)[0])
        assert all(not np.allclose(Rs[0], R, atol=1e-3) for R in Rs[1:])      # different mounting rotations


def test_donors_never_overlap_in_time(fold0):
    _, te, _ = fold0
    tr = fold0[0]
    b = O.OperatorBank("d3", "arms", te, tr, 0)
    for i in range(0, len(te), max(1, len(te) // 200)):
        for same in (True, False):
            d = b.donors(i, same)
            assert not (np.asarray(te.segment)[d] == te.segment[i]).any()
            assert (np.asarray(te.subject)[d] == te.subject[i]).all()


def _load_original_operators():
    """The operators file as it was before the WearGait-PD changes (kept by the test as a fixture copy)."""
    p = ROOT / "tests/fixtures/operators_before_weargait.py"
    if not p.exists():
        pytest.skip("fixture copy of the old operators file not present")
    spec = importlib.util.spec_from_file_location("ops_old", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_other_datasets_unchanged():
    """On MMAct-shaped data, the old and new operator code give byte-identical outputs."""
    old = _load_original_operators()

    class DS:                                         # a small synthetic MMAct-like split (phone layout)
        def __init__(self, n, subj, seed):
            r = np.random.default_rng(seed)
            self.x = {"phone": r.normal(size=(n, 40, 10)).astype(np.float32), "len": np.full(n, 40)}
            self.y = r.integers(0, 4, n); self.subject = np.full(n, subj); self.group = np.zeros(n, int)
            self.idx = np.arange(n) + 1000 * seed
            self.stats_file = None
    te, tr = DS(30, 1, 1), DS(60, 2, 2)
    stats = ROOT / "data/processed/norm_stats/zz_test_fold99.npz"
    np.savez(stats, phone_mean=np.zeros(10), phone_std=np.ones(10), train_subjects=np.array([2]))
    try:
        tr.stats_file = stats
        for op, lv in (("h_all", 2), ("h_rotate", 3), ("s_swap", 2), ("c_swap_same", 2)):
            a = O.OperatorBank("d2", "phone", te, tr, 99).apply(op, level=lv)
            b = old.OperatorBank("d2", "phone", te, tr, 99).apply(op, level=lv)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2]), op
    finally:
        stats.unlink()


def test_faults_on_groups(fold0):
    tr, te, _ = fold0
    fb = F.FaultBank("d3", "trunk", te, tr, 0)
    rows = slice(0, 30)
    x0 = phys(fb, fb.x[rows])
    # drift: each IMU's accelerometer offset grows linearly to 0.5 g (level 3) in its own direction
    xd, _, info = fb.apply_fault("f_drift", 3)
    d = phys(fb, xd[rows]) - x0
    ok = [k for k in range(30) if info[k]["clipped"] == 0]
    assert len(ok) >= 5
    for k in ok[:5]:
        dirs = []
        for acc, gyr in O.LAYOUTS[("d3", "trunk")].imus:
            last = d[k][-1][list(acc)]
            assert np.isclose(np.linalg.norm(last), 0.5 * F.G, rtol=1e-5)
            assert np.allclose(d[k][0][list(acc)], 0, atol=1e-6)
            assert np.isclose(np.linalg.norm(d[k][-1][list(gyr)]), np.deg2rad(20), rtol=1e-5)
            dirs.append(last / np.linalg.norm(last))
        assert not np.allclose(dirs[0], dirs[1])
    # gap: 25% of the 100 samples hold one value on every channel of the group
    xg, _, info = fb.apply_fault("f_gap", 2)
    for k in range(5):
        a, g = info[k]["gap_start"], info[k]["gap_len"]
        assert g == 25 and np.allclose(xg[k, a:a + g], xg[k, a:a + g][0])
    # shift: 0.2 s = 10 samples at 50 Hz, the whole group
    xs, _, info = fb.apply_fault("f_shift", 1)
    for k in range(5):
        s = info[k]["shift_samples"]
        assert abs(s) == 10
        src = fb.x[k]
        if s > 0:
            assert np.allclose(xs[k, 10:], src[:-10])
        else:
            assert np.allclose(xs[k, :-10], src[10:])


def test_realdrop_conditions(fold0):
    tr, te, dr = fold0
    from cert_conditions import build_realdrop_conditions, parts_for, conditions_suffix
    assert parts_for("realdrop") == ("test", "dropout") and conditions_suffix("realdrop") == "_realdrop"
    conds, gidx = build_realdrop_conditions("d3", te, tr, 0)
    assert list(conds) == ["clean", "r_fill:arms", "r_fill:trunk", "r_fill:legs"]
    for s in ("arms", "trunk", "legs"):
        repl, _, avail, att = conds[f"r_fill:{s}"]
        assert att == s and avail.all() and np.count_nonzero(repl[s]) == 0
    assert np.array_equal(gidx, te.idx)
    if len(dr):
        conds, gidx = build_realdrop_conditions("d3", dr, tr, 0)
        pats = sorted(set(dr.missing))
        assert list(conds) == [f"rd_fill:{p}" for p in pats]
        cover = np.zeros(len(dr), int)
        for name, (repl, _, avail, att) in conds.items():
            assert repl == {} and (np.asarray(dr.missing)[avail] == att).all()
            cover += avail
        assert (cover == 1).all()                    # every real-dropout clip under exactly one condition
