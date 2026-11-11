"""Fault library (docs/plan_faults_2026-11-11.md, step 1): known-answer tests on FAKE data (CPU, no model).

Each test checks one promise of the plan: the drift is an exact linear ramp of the stated size in the stated
units, the gap covers exactly the stated share and holds the right value, the shift moves the signal by exactly
the stated number of samples, padding and valid length never change, results are deterministic, clipping keeps
the training range, and faults appear only where the plan allows them.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import trustmi.operators as OPS_MOD
from trustmi.faults import (FaultBank, VideoFaultBank, DRIFT, GAP, SHIFT_S, FS, G, LEVELS, faults_for,
                            condition_name)


# ---------------------------------------------------------------- fake datasets in physical units
def _ds(x_phys, mu, sd, subjects, lens=None, trial=None, stats_file=None, extra=None):
    x = {k: (v - mu[k]) / sd[k] for k, v in x_phys.items()}
    if lens is not None:                                      # padding exactly 0 after normalisation
        valid = np.arange(next(iter(x.values())).shape[1])[None, :] < lens[:, None]
        x = {k: np.where(valid[..., None], v, 0.0) for k, v in x.items()}
        x["len"] = lens
    x.update(extra or {})
    n = len(subjects)
    return SimpleNamespace(x=x, y=np.arange(n) % 3, subject=np.asarray(subjects), group=np.zeros(n, int),
                           idx=np.arange(n) + 1000 * subjects[0], stats_file=stats_file,
                           trial=np.asarray(trial if trial is not None else [f"t{subjects[0]}_{i}" for i in range(n)]))


def _stats(tmp, name, mu, sd, train_subjects):
    f = tmp / f"{name}.npz"
    np.savez(f, **{f"{k}_mean": v for k, v in mu.items()}, **{f"{k}_std": v for k, v in sd.items()},
             train_subjects=np.asarray(train_subjects))
    return f


def utd(tmp, n=12, wide=True):
    """UTD-MHAD-like IMU: (N, 100, 6), accelerometer in g, gyroscope in deg/s; fixed length."""
    rng = np.random.default_rng(0)
    def make(k):
        acc = rng.normal(0, 0.5, (k, 100, 3)); gyr = rng.normal(0, 100, (k, 100, 3))
        return np.concatenate([acc, gyr], -1)
    tr, te = make(40), make(n)
    if wide:                                                  # training range far wider than any drift
        tr[0, 0, :3], tr[1, 0, :3] = -50, 50; tr[0, 0, 3:], tr[1, 0, 3:] = -5000, 5000
    mu, sd = {"imu": tr.reshape(-1, 6).mean(0)}, {"imu": tr.reshape(-1, 6).std(0)}
    skel_tr, skel_te = rng.normal(size=(40, 40, 20, 3)), rng.normal(size=(n, 40, 20, 3))
    mu["skel"], sd["skel"] = np.zeros((20, 3)), np.ones((20, 3))
    f = _stats(tmp, "utd", mu, sd, [1, 2])
    trd = _ds({"imu": tr, "skel": skel_tr}, mu, sd, [1] * 20 + [2] * 20, stats_file=f)
    ted = _ds({"imu": te, "skel": skel_te}, mu, sd, [3] * n, stats_file=f)
    return trd, ted


def mmact(tmp, n=10, monkeypatch=None, unit="deg"):
    """MMAct-like watch (3 ch, m/s^2) and phone (accel m/s^2, gyro rad/s, sin/cos pitch roll); padded to 750."""
    rng = np.random.default_rng(1)
    if monkeypatch is not None:
        monkeypatch.setattr(OPS_MOD, "_mmact_angle_unit", lambda: unit)
    T = 750
    def make(k, lens):
        w = rng.normal(0, 5, (k, T, 3))
        a = rng.normal(0, 4, (k, T, 3)); gy = rng.normal(0, 0.8, (k, T, 3))
        ang = rng.uniform(-60, 60, (k, T, 2)) * (1 if unit == "deg" else np.pi / 180)
        ph = np.concatenate([a, gy, np.sin(ang), np.cos(ang)], -1)
        return {"watch": w, "phone": ph}, lens
    tr, ltr = make(40, rng.integers(50, 751, 40))
    tr["watch"][0, 0], tr["watch"][1, 0] = -200, 200
    tr["phone"][0, 0, :6], tr["phone"][1, 0, :6] = -200, 200
    ltr[:2] = 750
    te, lte = make(n, np.r_[50, 60, 750, rng.integers(51, 750, n - 3)])
    def stats(k):
        valid = np.arange(T)[None, :] < ltr[:, None]
        v = tr[k][valid]
        return v.mean(0), v.std(0) + 1e-6
    mu, sd = {}, {}
    for k in ("watch", "phone"):
        mu[k], sd[k] = stats(k)
    f = _stats(tmp, "mmact", mu, sd, list(range(1, 11)))
    trd = _ds(tr, mu, sd, np.repeat(np.arange(1, 11), 4), lens=ltr, stats_file=f)
    ted = _ds(te, mu, sd, [12] * n, lens=lte, stats_file=f)
    return trd, ted


F32 = 4e-7          # float32 relative resolution (2^-23 = 1.2e-7) with margin; fault outputs are float32


def phys(bank, x, i):
    n = bank.lens[i]
    return x.reshape(bank.N, bank.T, bank.C)[i, :n].astype(np.float64) * bank.sd + bank.mu


# ---------------------------------------------------------------- drift
@pytest.mark.parametrize("level", LEVELS)
def test_drift_is_an_exact_linear_ramp_of_the_stated_size_utd(tmp_path, level):
    tr, te = utd(tmp_path)
    b = FaultBank("d1v2", "imu", te, tr, fold=0)
    x, avail, info = b.apply_fault("f_drift", level)
    assert avail.all() and x.shape == te.x["imu"].shape
    for i in range(b.N):
        d = phys(b, x, i) - phys(b, b.x, i)
        assert info[i]["clipped"] == 0
        n = len(d); ramp = np.arange(n) / (n - 1)
        for ch, size in (((0, 1, 2), DRIFT[level]["acc_g"]), ((3, 4, 5), DRIFT[level]["gyro_dps"])):
            v = d[:, ch]
            tol = F32 * np.abs(phys(b, b.x, i)[:, ch]).max()        # what float32 storage can resolve
            assert np.allclose(v[0], 0, atol=tol)
            assert np.isclose(np.linalg.norm(v[-1]), size, atol=2 * tol)          # stated length, in g / deg/s
            assert np.allclose(v, ramp[:, None] * v[-1], atol=2 * tol)            # straight line in time


def test_drift_units_mmact(tmp_path, monkeypatch):
    tr, te = mmact(tmp_path, monkeypatch=monkeypatch, unit="deg")
    for sensor in ("watch", "phone"):
        b = FaultBank("d2v", sensor, te, tr, fold=0)
        x, _, info = b.apply_fault("f_drift", 1)
        for i in range(b.N):
            if info[i]["clipped"]:
                continue
            d = phys(b, x, i) - phys(b, b.x, i)
            tol = F32 * np.abs(phys(b, b.x, i)).max()
            assert np.isclose(np.linalg.norm(d[-1, :3]), 0.05 * G, atol=2 * tol)      # 0.05 g in m/s^2
            if sensor == "phone":
                assert np.isclose(np.linalg.norm(d[-1, 3:6]), np.deg2rad(1.0), atol=2 * tol)  # 1 deg/s in rad/s
            assert np.all(x.reshape(b.N, b.T, b.C)[i, b.lens[i]:] == 0)               # padding untouched


@pytest.mark.parametrize("unit", ["deg", "rad"])
def test_drift_of_pitch_and_roll(tmp_path, monkeypatch, unit):
    """The loader feeds the RAW angle numbers to sin / cos, so the offset is added in the data's own unit
    (as the harmless h_angle operator does): 30 when the data are in degrees, 30 deg in radians otherwise."""
    tr, te = mmact(tmp_path, monkeypatch=monkeypatch, unit=unit)
    size = 30.0 if unit == "deg" else np.deg2rad(30.0)
    wrap = lambda a: np.angle(np.exp(1j * a))
    b = FaultBank("d2v", "phone", te, tr, fold=0)
    x, _, _ = b.apply_fault("f_drift", 3)
    for i in range(b.N):
        p0, p1 = phys(b, b.x, i), phys(b, x, i)
        n = len(p0)
        dd = wrap(np.arctan2(p1[:, 6:8], p1[:, 8:10]) - np.arctan2(p0[:, 6:8], p0[:, 8:10]))
        u = dd[1] * (n - 1) / size                                 # first step is small: no wrap-around
        assert np.isclose(np.linalg.norm(u), 1, atol=1e-3)        # a 2-D direction of length one
        ramp = np.arange(n) / (n - 1)
        assert np.allclose(wrap(dd - ramp[:, None] * size * u), 0, atol=1e-4)   # exact ramp, end = stated size
        assert np.allclose(p1[:, 6:8] ** 2 + p1[:, 8:10] ** 2, 1, atol=1e-5)


def test_drift_is_clipped_to_the_training_range(tmp_path):
    tr, te = utd(tmp_path, wide=False)
    b = FaultBank("d1v2", "imu", te, tr, fold=0)
    x, _, info = b.apply_fault("f_drift", 3)
    p = x.reshape(b.N, b.T, b.C).astype(np.float64) * b.sd + b.mu
    assert np.all(p >= b.lo - 1e-4) and np.all(p <= b.hi + 1e-4)
    assert sum(i["clipped"] for i in info) > 0


# ---------------------------------------------------------------- gap
@pytest.mark.parametrize("level", LEVELS)
def test_gap_covers_the_stated_share_and_holds_the_last_value(tmp_path, monkeypatch, level):
    tr, te = mmact(tmp_path, monkeypatch=monkeypatch)
    b = FaultBank("d2v", "watch", te, tr, fold=0)
    x, _, info = b.apply_fault("f_gap", level)
    x = x.reshape(b.N, b.T, b.C)
    for i in range(b.N):
        n, a, g = b.lens[i], info[i]["gap_start"], info[i]["gap_len"]
        assert g == round(GAP[level] * n) and 0 <= a and a + g <= n
        hold = b.x[i, a - 1] if a > 0 else b.x[i, a + g]
        assert np.allclose(x[i, a:a + g], hold)
        keep = np.r_[0:a, a + g:n]
        assert np.allclose(x[i, keep], b.x[i, keep])
        assert np.all(x[i, n:] == 0)


def test_gap_on_skeleton_holds_a_whole_frame(tmp_path):
    tr, te = utd(tmp_path)
    b = FaultBank("d1v2", "skel", te, tr, fold=0)
    x, _, info = b.apply_fault("f_gap", 2)
    assert x.shape == te.x["skel"].shape
    for i in range(b.N):
        a, g = info[i]["gap_start"], info[i]["gap_len"]
        assert g == 10                                             # 25% of 40 frames
        hold = te.x["skel"][i, a - 1] if a > 0 else te.x["skel"][i, a + g]
        assert np.allclose(x[i, a:a + g], hold)                   # all 20 joints frozen


# ---------------------------------------------------------------- shift
@pytest.mark.parametrize("level", LEVELS)
def test_shift_moves_by_exactly_the_stated_samples(tmp_path, monkeypatch, level):
    tr, te = mmact(tmp_path, monkeypatch=monkeypatch)
    b = FaultBank("d2v", "phone", te, tr, fold=0)
    x, _, info = b.apply_fault("f_shift", level)
    x = x.reshape(b.N, b.T, b.C)
    k0 = int(round(SHIFT_S[level] * FS["d2"]))
    for i in range(b.N):
        n, s = b.lens[i], info[i]["shift_samples"]
        k = abs(s)
        assert k == min(k0, n - 1)
        v = b.x[i, :n]
        if s > 0:
            assert np.allclose(x[i, k:n], v[:n - k]) and np.allclose(x[i, :k], v[0])
        else:
            assert np.allclose(x[i, :n - k], v[k:]) and np.allclose(x[i, n - k:n], v[n - 1])
        assert np.all(x[i, n:] == 0)
        assert info[i]["shift_over_half"] == (k > n / 2)


# ---------------------------------------------------------------- common rules
def test_deterministic_and_padding_never_changes(tmp_path, monkeypatch):
    tr, te = mmact(tmp_path, monkeypatch=monkeypatch)
    for sensor in ("watch", "phone"):
        b = FaultBank("d2v", sensor, te, tr, fold=0)
        for fault in faults_for("d2v", sensor):
            for lv in LEVELS:
                a, _, ia = b.apply_fault(fault, lv)
                c, _, ic = FaultBank("d2v", sensor, te, tr, fold=0).apply_fault(fault, lv)
                assert np.array_equal(a, c) and ia == ic
                assert np.all(a.reshape(b.N, b.T, b.C)[np.arange(b.T)[None, :] >= b.lens[:, None]] == 0)
                assert not np.array_equal(a, b.x.astype(np.float32))      # the fault changes something
    # different levels and different faults draw differently
    b = FaultBank("d2v", "watch", te, tr, fold=0)
    starts = [[i["gap_start"] for i in b.apply_fault("f_gap", lv)[2]] for lv in LEVELS]
    assert starts[0] != starts[1] or starts[1] != starts[2]


def test_faults_only_where_the_plan_allows(tmp_path, monkeypatch):
    tr, te = utd(tmp_path)
    with pytest.raises(ValueError):
        FaultBank("d1v2", "imu", te, tr, fold=0).apply_fault("f_shift", 1)     # no timing faults on UTD-MHAD
    with pytest.raises(ValueError):
        FaultBank("d1v2", "skel", te, tr, fold=0).apply_fault("f_drift", 1)    # Kinect has no drift
    with pytest.raises(ValueError):
        FaultBank("d1", "imu", te, tr, fold=0)                                 # conference datasets: none
    assert faults_for("d1v2", "video") == ("f_gap",) and faults_for("d2v", "video") == ("f_gap", "f_shift")
    assert condition_name("f_gap", 2, "watch") == "f_gap:L2:watch"


# ---------------------------------------------------------------- video lookup
def test_video_fault_bank_normalises_and_checks_the_clean_tokens(tmp_path):
    rng = np.random.default_rng(3)
    trials = np.array([f"c{i}" for i in range(8)])
    raw = {"clean": rng.normal(size=(8, 4, 512)), "f_gap_L1": rng.normal(size=(8, 4, 512)),
           "f_gap_L2": rng.normal(size=(8, 4, 512)), "f_gap_L3": rng.normal(size=(8, 4, 512))}
    mu, sd = rng.normal(size=512), rng.uniform(0.5, 2, 512)
    st = _stats(tmp_path, "v", {"video": mu}, {"video": sd}, [1])
    ff = tmp_path / "vf.npz"
    np.savez(ff, trial=trials[::-1], **{k: v[::-1] for k, v in raw.items()})    # file order differs on purpose
    rows = np.array([5, 2, 7])
    te = SimpleNamespace(x={"video": ((raw["clean"][rows] - mu) / sd).astype(np.float32)}, trial=trials[rows],
                         subject=np.array([3, 3, 3]))
    tr = SimpleNamespace(subject=np.array([1, 1]), stats_file=st)
    vb = VideoFaultBank("d1v2", te, tr, fold=0, path=ff)
    x, avail, _ = vb.apply_fault("f_gap", 2)
    assert avail.all() and np.allclose(x, (raw["f_gap_L2"][rows] - mu) / sd, atol=1e-5)
    with pytest.raises(ValueError):
        vb.apply_fault("f_shift", 1)                                              # not planned on UTD-MHAD
    noisy = te.x["video"].copy(); noisy[0, 0, :5] += 0.05                          # bf16-sized noise: accepted
    VideoFaultBank("d1v2", SimpleNamespace(**{**vars(te), "x": {"video": noisy}}), tr, fold=0, path=ff)
    te.x["video"] = te.x["video"] + 1.0                                           # wrong clips -> must stop
    with pytest.raises(RuntimeError):
        VideoFaultBank("d1v2", te, tr, fold=0, path=ff)
    bad = (te.x["video"] - 1.0).copy(); bad[0, 0, 0] += 0.5                       # one large outlier -> must stop
    with pytest.raises(RuntimeError):
        VideoFaultBank("d1v2", SimpleNamespace(**{**vars(te), "x": {"video": bad}}), tr, fold=0, path=ff)


# ---------------------------------------------------------------- plan step 2: the fault bench
def test_fault_conditions_have_the_planned_names_and_format(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    from cert_conditions import build_fault_conditions, conditions_for, conditions_suffix
    for d, (tr, te), expected in (("d1v2", utd(tmp_path), 1 + 2 * 3 + 3),               # IMU drift+gap, skeleton gap
                                  ("d2v", mmact(tmp_path, monkeypatch=monkeypatch), 1 + 9 + 9)):   # watch, phone
        conds, gidx = build_fault_conditions(d, te, tr, fold=0, video=False)
        assert len(conds) == expected and list(conds)[0] == "clean"
        assert np.array_equal(gidx, te.idx)
        for name, (repl, donor_y, avail, attacked) in conds.items():
            if name == "clean":
                assert repl == {} and attacked == ""
                continue
            fault, lv, sensor = name.split(":")
            assert fault in faults_for(d, sensor) and lv in ("L1", "L2", "L3") and attacked == sensor
            assert list(repl) == [sensor] and repl[sensor].shape == np.asarray(te.x[sensor]).shape
            assert repl[sensor].dtype == np.float32 and avail.all() and (donor_y == -1).all()
        again, _ = build_fault_conditions(d, te, tr, fold=0, video=False)             # deterministic
        assert all(np.array_equal(conds[k][0].get(s, 0), again[k][0].get(s, 0)) for k in conds for s in conds[k][0])
        assert "f_shift:L1:imu" not in conds
    assert conditions_suffix("lies") == "" and conditions_suffix("faults") == "_faults"
    assert conditions_suffix("faults", video=False) == "_faults-novideo"
    with pytest.raises(ValueError):
        conditions_for("other", "d1v2", te, tr, 0, [], [])
