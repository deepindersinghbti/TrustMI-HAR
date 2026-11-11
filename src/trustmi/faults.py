"""Real sensor faults for the journal paper: drift, partial dropout (gap), timing error (shift).

Fixed in advance in docs/plan_faults_2026-11-11.md (approved and committed before any fault result).
Built ON the counterfactual operator bank (trustmi.operators.OperatorBank), which is NOT edited: the faults reuse
its physical units, channel roles, training-range clipping and padding handling, so the conference code cannot
change. Same rules as the swaps: only valid samples change, padding stays exactly 0, the valid length never
changes, and every random draw is keyed on (seed, fault, level, clip), so results are exactly reproducible.

  f_drift   a bias growing linearly from 0 at the first valid sample to its full size at the last one.
            3-axis sensors: one random direction per clip (a bias VECTOR of the stated length); accelerometer and
            gyroscope get independent directions. Phone pitch / roll: a random 2-D direction of the stated length.
            Computed in physical units, then clipped to the range seen in training (a real sensor saturates).
  f_gap     one contiguous stretch covering the stated share of the valid samples, at a random position fully
            inside the clip, filled by holding the last value before it (a gap at the very start holds the
            first value after it). Holding a value is the same in physical and normalised units.
  f_shift   the sensor moved earlier or later (random) by the stated time; samples pushed out are lost, the
            vacated end holds the edge value, the valid length is unchanged. MMAct only (plan, section 4).

Strengths (plan, section 3), swept and never tuned:
  drift   accelerometer 0.05 / 0.2 / 0.5 g, gyroscope 1 / 5 / 20 deg/s, pitch / roll 2 / 10 / 30 deg
  gap     10 / 25 / 50 % of the valid samples
  shift   0.2 / 0.5 / 1.0 s
Units of the data (docs/data_audit_*.md): UTD-MHAD IMU accelerometer in g, gyroscope in deg/s; MMAct watch and
phone accelerometer in m/s^2, phone gyroscope in rad/s; MMAct pitch / roll unit detected from the data.
Video faults are applied to the decoded frames by scripts/extract_video_faults.py (plan, section 5, option A);
VideoFaultBank only looks the resulting tokens up and normalises them with the fold's training statistics.
"""
from pathlib import Path
import numpy as np
from trustmi.operators import OperatorBank, BASE_DATASET, _stats_for

ROOT = Path(__file__).resolve().parents[2]
G = 9.80665
FAULTS = ("f_drift", "f_gap", "f_shift")
LEVELS = (1, 2, 3)
DRIFT = {1: dict(acc_g=0.05, gyro_dps=1.0, ang_deg=2.0),
         2: dict(acc_g=0.20, gyro_dps=5.0, ang_deg=10.0),
         3: dict(acc_g=0.50, gyro_dps=20.0, ang_deg=30.0)}
GAP = {1: 0.10, 2: 0.25, 3: 0.50}
SHIFT_S = {1: 0.2, 2: 0.5, 3: 1.0}
UNITS = {("d1", "imu"): dict(accel="g", gyro="deg/s"),
         ("d2", "watch"): dict(accel="m/s2"),
         ("d2", "phone"): dict(accel="m/s2", gyro="rad/s")}
FS = {"d2": 50}                       # MMAct sensors, Hz. UTD-MHAD has no timing faults (plan, section 4)
APPLIES = {                           # plan, section 4
    ("d1v2", "imu"): ("f_drift", "f_gap"), ("d1v2", "skel"): ("f_gap",), ("d1v2", "video"): ("f_gap",),
    ("d2v", "watch"): FAULTS, ("d2v", "phone"): FAULTS, ("d2v", "video"): ("f_gap", "f_shift"),
}
FAULT_SALT = 100                      # fault ids 100.. keep the draws apart from the operators' (0..9)
VIDEO_FAULTS = {"d1v2": ROOT / "data/features/video_r2plus1d18_v2_faults.npz",
                "d2v": ROOT / "data/features/mmact_video_r2plus1d18_v2_faults.npz"}


def condition_name(fault, level, sensor):
    return f"{fault}:L{level}:{sensor}"


def faults_for(dataset, sensor):
    return APPLIES.get((dataset, sensor), ())


def _unit_vector(rng, k):
    v = rng.normal(size=k)
    return v / np.linalg.norm(v)


class FaultBank(OperatorBank):
    """Faults for one (journal dataset, motion sensor or skeleton) on one evaluation split."""

    def __init__(self, dataset, modality, ds, train_ds, fold, seed=0):
        if modality == "video":
            raise ValueError("video faults come from VideoFaultBank (frame-level features)")
        if not faults_for(dataset, modality):
            raise ValueError(f"no faults planned for {dataset}/{modality}")
        super().__init__(dataset, modality, ds, train_ds, fold, seed)
        self.base = BASE_DATASET.get(dataset, dataset)

    def _frng(self, fault, level, i):
        return np.random.default_rng([self.seed, FAULT_SALT + FAULTS.index(fault), level, int(self.gidx[i])])

    # ---------------------------------------------------------------- the three faults, one clip
    def _drift(self, i, level, rng):
        L, P, U = self.layout, DRIFT[level], UNITS[(self.base, self.modality)]
        p, n = self._phys(i)
        p = p.copy()
        ramp = np.arange(n) / max(n - 1, 1)                                  # 0 ... 1 over the valid samples
        info = {}
        if L.accel:
            size = P["acc_g"] * (1.0 if U["accel"] == "g" else G)
            u = _unit_vector(rng, 3)
            p[:, list(L.accel)] += ramp[:, None] * size * u
            info["drift_accel"] = size
        if L.gyro:
            size = P["gyro_dps"] if U["gyro"] == "deg/s" else np.deg2rad(P["gyro_dps"])
            u = _unit_vector(rng, 3)
            p[:, list(L.gyro)] += ramp[:, None] * size * u
            info["drift_gyro"] = size
        phys_ch = list(L.accel) + list(L.gyro)
        before = p[:, phys_ch].copy()
        p[:, phys_ch] = np.clip(p[:, phys_ch], self.lo[phys_ch], self.hi[phys_ch])
        info.update(clipped=int((before != p[:, phys_ch]).sum()), n_values=n * len(phys_ch))
        if L.ang_sin:
            s, c = list(L.ang_sin), list(L.ang_cos)
            size = np.deg2rad(P["ang_deg"]) if self.angle_unit == "rad" else P["ang_deg"]
            u = _unit_vector(rng, len(s))
            ang = np.arctan2(p[:, s], p[:, c]) + ramp[:, None] * size * u
            p[:, s], p[:, c] = np.sin(ang), np.cos(ang)
            info["drift_angle"] = size
        return self._back(i, p, n), info

    def _gap(self, i, level, rng):
        n = int(self.lens[i])
        g = min(max(1, int(round(GAP[level] * n))), n - 1)                   # at least one sample is kept
        a = int(rng.integers(0, n - g + 1))
        out = np.zeros((self.T, self.C))
        out[:n] = self.x[i, :n]
        out[a:a + g] = self.x[i, a - 1] if a > 0 else self.x[i, a + g]
        return out, dict(gap_start=a, gap_len=g)

    def _shift(self, i, level, rng):
        if self.base not in FS:
            raise ValueError(f"no timing faults on {self.dataset} (plan, section 4)")
        n = int(self.lens[i])
        k = min(int(round(SHIFT_S[level] * FS[self.base])), n - 1)
        d = int(rng.choice([-1, 1]))                                         # +1 later (delay), -1 earlier
        v = self.x[i, :n]
        out = np.zeros((self.T, self.C))
        if d > 0:
            out[k:n], out[:k] = v[:n - k], v[0]
        else:
            out[:n - k], out[n - k:n] = v[k:], v[n - 1]
        return out, dict(shift_samples=k * d, shift_over_half=bool(k > n / 2))

    # ---------------------------------------------------------------- public
    def apply_fault(self, fault, level):
        """Every clip with this fault. Returns (x_new float32 in the sensor's own shape, available (N,) bool,
        info list). All clips are available (no rule can make a fault impossible)."""
        if fault not in faults_for(self.dataset, self.modality):
            raise ValueError(f"{fault} is not planned for {self.dataset}/{self.modality}")
        fn = {"f_drift": self._drift, "f_gap": self._gap, "f_shift": self._shift}[fault]
        x_new, infos = np.zeros_like(self.x), []
        for i in range(self.N):
            x_new[i], info = fn(i, level, self._frng(fault, level, i))
            infos.append(dict(clip=int(self.gidx[i]), fault=fault, level=level, **info))
        return x_new.reshape(self.shape).astype(np.float32), np.ones(self.N, bool), infos


class VideoFaultBank:
    """Video faults made at frame level (scripts/extract_video_faults.py). Looks the tokens up by trial NAME,
    normalises them with the fold's training statistics exactly as the loaders do, and checks that the stored
    clean tokens give back the loader's video input."""

    # Recomputing the clean clip on the GPU differs from the stored features only by bf16 rounding: measured on all
    # 861 UTD-MHAD videos (11 Nov 2026), mean 0.0000, 99.9% 0.011, max 0.05 feature spreads, while the mildest gap
    # changes the features by mean 0.13, max 3.7. A wrong clip or file is off by about one whole spread on average.
    CLEAN_TOL_MEAN, CLEAN_TOL_MAX = 0.002, 0.1

    def __init__(self, dataset, ds, train_ds, fold, path=None):
        self.dataset, self.fold = dataset, fold
        f = np.load(path or VIDEO_FAULTS[dataset])
        pos = {t: r for r, t in enumerate(f["trial"])}
        missing = [t for t in ds.trial if t not in pos]
        if missing:
            raise RuntimeError(f"{len(missing)} clips have no fault video features (e.g. {missing[0]})")
        rows = np.array([pos[t] for t in ds.trial])
        mu, sd, _ = _stats_for("video", fold, np.unique(np.asarray(train_ds.subject)),
                               getattr(train_ds, "stats_file", None))
        self._norm = lambda a: ((a[rows] - mu) / sd).astype(np.float32)
        self._f = f
        clean = self._norm(f["clean"])
        d = np.abs(clean - np.asarray(ds.x["video"], dtype=np.float32))
        mean, mx = float(d.mean()), float(d.max())
        if mean > self.CLEAN_TOL_MEAN or mx > self.CLEAN_TOL_MAX:
            raise RuntimeError(f"clean fault-file tokens differ from the loader's video (mean {mean:.3g}, max {mx:.3g}; "
                               f"allowed {self.CLEAN_TOL_MEAN}, {self.CLEAN_TOL_MAX}): stop and paste this")
        self.N, self.clean_dev = len(rows), (mean, mx)

    def apply_fault(self, fault, level):
        if fault not in faults_for(self.dataset, "video"):
            raise ValueError(f"{fault} is not planned for {self.dataset}/video")
        x = self._norm(self._f[f"{fault}_L{level}"])
        return x, np.ones(self.N, bool), [dict(fault=fault, level=level)] * self.N
