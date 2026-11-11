"""Tests for scripts/build_weargait.py on synthetic recordings whose answers are known in advance.
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_weargait_build.py -q
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_weargait as B                                                     # noqa: E402

ALL_CH = [c for s in B.SENSORS for c in B.channels(s)]


def recording(tmp_path, segments, fill=None, drop_locs=(), name="HC999_SelfPace.csv"):
    """segments: list of (label, seconds). fill(t, channel_index) -> values (default: smooth random)."""
    labels = np.concatenate([[lab] * int(round(sec * 100)) for lab, sec in segments])
    n = len(labels)
    t = np.arange(n) / 100
    rng = np.random.default_rng(0)
    cols = {"Time": [f"{v:g} sec" for v in t], "GeneralEvent": labels}
    for i, c in enumerate(ALL_CH):
        if any(c.startswith(loc + "_") for loc in drop_locs):
            continue
        cols[c] = fill(t, i) if fill else np.cumsum(rng.normal(0, 0.01, n))
    p = tmp_path / name
    pd.DataFrame(cols).to_csv(p, index=False)
    return p


def test_clip_rules(tmp_path):
    """Clips: 2 s, 1 s step, inside one segment, at most 8; balance -> Standing; dropped labels skipped."""
    p = recording(tmp_path, [("Walk", 4.0), ("Turn", 1.5), ("EO_FeetTogether", 15.0), ("SitToStand", 3.0),
                             ("unlabeled", 5.0), ("Sitting", 2.0)])
    clips, X, info = B.scan(p, "control")
    C = pd.DataFrame(clips)
    assert C.groupby("cls").size().to_dict() == {"Walk": 3, "Standing": 8, "Sitting": 1}   # Turn 1.5 s too short
    assert (C[C.cls == "Standing"].label == "EO_FeetTogether").all()
    seg_bounds = {0: (0, 200), 2: (275, 1025), 5: (1425, 1525)}       # 50 Hz sample ranges of kept segments
    for _, c in C.iterrows():
        a, b = seg_bounds[c.segment]
        assert a <= c.start and c.start + B.T <= b
    assert list(C[C.cls == "Standing"].start) == [275 + 50 * j for j in range(8)]
    assert X["arms"].shape == (12, 100, 12) and X["trunk"].shape[2] == 18 and X["legs"].shape[2] == 48
    assert C.clean.all() and info["n_time_jumps"] == 0


def test_filter_keeps_slow_and_removes_fast(tmp_path):
    """A 2 Hz signal passes almost unchanged; a 45 Hz signal (above 25 Hz, would alias) is removed."""
    k = np.arange(2000)
    slow, fast = np.sin(2 * np.pi * 2 * k / 100), np.sin(2 * np.pi * 45 * k / 100)
    out = B.filter_and_decimate(np.stack([slow, fast], 1))
    mid = slice(50, -50)
    assert np.abs(out[mid, 0] - slow[::2][mid]).max() < 0.01
    assert np.abs(out[mid, 1]).max() < 0.01


def test_missing_values_stay_missing_and_mark_clips(tmp_path):
    """A gap in one leg channel: those samples stay NaN, the clips touching the gap (with the filter's reach)
    are real-dropout clips for 'legs' only, and every other clip is clean and NaN-free."""
    gap = (500, 600)                                       # 100 Hz samples 5.0-6.0 s (inside the 8 clips)
    leg0 = len(B.channels("arms")) + len(B.channels("trunk"))

    def fill(t, i):
        v = np.sin(t + i)
        if i == leg0:
            v = v.copy(); v[gap[0]:gap[1]] = np.nan
        return v
    p = recording(tmp_path, [("Walk", 30.0)], fill=fill)
    clips, X, _ = B.scan(p, "PD")
    C = pd.DataFrame(clips)
    lo, hi = (gap[0] - B.REACH) / 2, (gap[1] + B.REACH) / 2           # widened gap at 50 Hz
    touch = (C.start < hi) & (C.start + B.T > lo)
    assert touch.sum() >= 2 and (C.clean == ~touch).all()
    assert (C[~C.clean].missing == "legs").all()
    for s in B.SENSORS:
        assert not np.isnan(X[s][C.clean.to_numpy()]).any()
    assert np.isnan(X["arms"]).sum() == 0 and np.isnan(X["legs"]).sum() > 0


def test_absent_imu_is_missing_everywhere(tmp_path):
    p = recording(tmp_path, [("Walk", 5.0)], drop_locs=("Forehead",))
    clips, X, info = B.scan(p, "PD")
    C = pd.DataFrame(clips)
    assert (~C.clean).all() and (C.missing == "trunk").all()
    assert "trunk:Forehead" in info["absent"]
    assert np.isnan(X["trunk"][:, :, :6]).all() and not np.isnan(X["trunk"][:, :, 6:]).any()


def test_deterministic(tmp_path):
    p = recording(tmp_path, [("Walk", 6.0), ("Standing", 4.0)])
    a, b = B.scan(p, "PD"), B.scan(p, "PD")
    for s in B.SENSORS:
        assert np.array_equal(a[1][s], b[1][s], equal_nan=True)
    assert a[0] == b[0]
