"""Addition 1 (docs/plan_additions_2026-11-14.md, Checks 3): stuck sensor and saturation, known-answer tests on
FAKE data (CPU, no model), with the fake datasets of tests/test_faults.py.
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_faults2.py -q
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "scripts"))
from test_faults import utd, mmact, F32                                         # noqa: E402
from trustmi.faults import G                                                    # noqa: E402
from trustmi.faults2 import (FaultBank2, STUCK, SAT, LEVELS, APPLIES2, stuck_span, faults2_for)  # noqa: E402


def phys(bank, x):
    """Normalised (N, T, C) -> physical units, with the bank's training statistics."""
    return x.reshape(bank.N, bank.T, bank.C) * bank.sd + bank.mu


# ---------------------------------------------------------------- stuck
@pytest.mark.parametrize("level", LEVELS)
def test_stuck_freezes_at_the_onset_until_the_end(tmp_path, monkeypatch, level):
    tr, te = mmact(tmp_path, monkeypatch=monkeypatch)
    for s in ("watch", "phone"):
        b = FaultBank2("d2v", s, te, tr, fold=0)
        xn, avail, info = b.apply_fault("f_stuck", level)
        xn = xn.reshape(b.N, b.T, b.C)
        assert avail.all() and xn.dtype == np.float32
        for i in range(b.N):
            n = int(b.lens[i])
            a, g = stuck_span(STUCK[level], n)
            assert a + g == n and g == min(max(1, round(STUCK[level] * n)), n) and info[i]["stuck_start"] == a
            assert np.allclose(xn[i, a:n], b.x[i, a], atol=F32 * (1 + np.abs(b.x[i, a]).max()))  # frozen value
            assert np.array_equal(xn[i, :a], b.x[i, :a].astype(np.float32))                    # before: unchanged
            assert (xn[i, n:] == 0).all()                                                      # padding stays 0
        if level == 3:
            assert all(r["stuck_start"] == 0 for r in info)                                    # the whole clip


def test_stuck_on_skeleton_holds_a_whole_frame(tmp_path):
    tr, te = utd(tmp_path)
    b = FaultBank2("d1v2", "skel", te, tr, fold=0)
    xn, _, _ = b.apply_fault("f_stuck", 2)
    xn = xn.reshape(b.N, b.T, b.C)
    a, _ = stuck_span(STUCK[2], b.T)
    assert a == b.T // 2
    assert np.allclose(xn[:, a:], xn[:, a:a + 1])
    with pytest.raises(ValueError):
        b.apply_fault("f_sat", 1)                                                  # no saturation on the skeleton


def test_stuck_span_examples():
    assert stuck_span(0.25, 100) == (75, 25)
    assert stuck_span(0.5, 101) == (51, 50)                  # Python rounds 50.5 to 50 (half to even), as the gap fault
    assert stuck_span(1.0, 37) == (0, 37) and stuck_span(0.25, 2) == (1, 1) and stuck_span(0.25, 1) == (0, 1)


# ---------------------------------------------------------------- saturation
@pytest.mark.parametrize("level", LEVELS)
def test_saturation_clips_at_the_full_scale_in_the_right_units_utd(tmp_path, level):
    tr, te = utd(tmp_path)                                    # accelerometer in g, gyroscope in deg/s
    te.x["imu"] = te.x["imu"] * 40                            # make some readings exceed every range
    b = FaultBank2("d1v2", "imu", te, tr, fold=0)
    xn, _, info = b.apply_fault("f_sat", level)
    p_in, p_out = phys(b, b.x), phys(b, xn)
    ra, rg = SAT[level]["acc_g"], SAT[level]["gyro_dps"]
    assert np.abs(p_out[..., :3]).max() <= ra * (1 + 1e-5) and np.abs(p_out[..., 3:]).max() <= rg * (1 + 1e-5)
    clipped_a = np.abs(p_in[..., :3]) > ra
    assert np.allclose(np.abs(p_out[..., :3][clipped_a]), ra, rtol=1e-5)          # clipped values sit on the range
    keep = np.abs(p_in) <= np.r_[[ra] * 3, [rg] * 3]
    xin = b.x.reshape(xn.reshape(b.N, b.T, b.C).shape)
    assert np.array_equal(xn.reshape(b.N, b.T, b.C)[keep], xin[keep].astype(np.float32))   # others: exactly as given
    assert sum(r["clipped"] for r in info) == int((~keep).sum())


def test_saturation_mmact_units_and_angles_untouched(tmp_path, monkeypatch):
    tr, te = mmact(tmp_path, monkeypatch=monkeypatch)        # m/s^2 and rad/s; phone pitch / roll as sin, cos
    for s in ("watch", "phone"):
        te.x[s] = te.x[s] * 1.0
    b = FaultBank2("d2v", "phone", te, tr, fold=0)
    big = b.x.copy()
    big[..., :6] *= 30                                        # accelerometer and gyroscope far out of range
    b.x = big
    xn, _, _ = b.apply_fault("f_sat", 3)
    xn = xn.reshape(b.N, b.T, b.C)
    p = phys(b, xn)
    valid = np.arange(b.T)[None, :] < b.lens[:, None]
    assert np.isclose(np.abs(p[..., :3][valid]).max(), 2 * G, rtol=1e-5)       # clipped AT +/-2 g, in m/s^2
    assert np.isclose(np.abs(p[..., 3:6][valid]).max(), np.deg2rad(250), rtol=1e-5)   # AT +/-250 deg/s, in rad/s
    assert np.array_equal(xn[..., 6:], b.x[..., 6:].astype(np.float32))         # pitch / roll exactly unchanged
    assert (xn[~valid] == 0).all()                                               # padding stays 0
    w = FaultBank2("d2v", "watch", te, tr, fold=0)
    w.x = w.x * 30
    xw = phys(w, w.apply_fault("f_sat", 1)[0])
    assert np.isclose(np.abs(xw[valid]).max(), 8 * G, rtol=1e-5)                 # watch: AT +/-8 g, in m/s^2


def test_saturation_level1_can_clip_nothing(tmp_path):
    tr, te = utd(tmp_path)                                    # ordinary readings: far inside +/-8 g, +/-1000 deg/s
    b = FaultBank2("d1v2", "imu", te, tr, fold=0)
    xn, _, info = b.apply_fault("f_sat", 1)
    assert sum(r["clipped"] for r in info) == 0
    assert np.array_equal(xn.reshape(b.N, b.T, b.C), b.x.astype(np.float32))    # nothing clipped = input exactly


# ---------------------------------------------------------------- plan
def test_deterministic_and_only_where_planned(tmp_path, monkeypatch):
    tr, te = mmact(tmp_path, monkeypatch=monkeypatch)
    b = FaultBank2("d2v", "watch", te, tr, fold=0)
    for f in ("f_stuck", "f_sat"):
        assert np.array_equal(b.apply_fault(f, 2)[0], b.apply_fault(f, 2)[0])
    assert faults2_for("d1v2", "skel") == ("f_stuck",)
    assert all(set(v) <= {"f_stuck", "f_sat"} for v in APPLIES2.values())
    assert {s for d, s in APPLIES2 if d == "d3"} == {"arms", "trunk", "legs"}
    with pytest.raises(ValueError):
        FaultBank2("d1v2", "video", te, tr, fold=0)


def test_video_stuck_map_and_names():
    import importlib
    try:
        EV = importlib.import_module("extract_video_faults2")
    except Exception as e:                                    # needs torch / cv2 / torchvision
        pytest.skip(f"video script not importable here: {e}")
    for lv, onset in ((1, 24), (2, 16), (3, 0)):
        m, a = EV.stuck_map(lv)
        assert a == onset and (m[a:] == a).all() and (m[:a] == np.arange(a)).all()
    assert [EV.key(*f) for f in EV.FAULTS2] == ["f_stuck_L1", "f_stuck_L2", "f_stuck_L3",
                                                "f_sat_L1", "f_sat_L2", "f_sat_L3"]
