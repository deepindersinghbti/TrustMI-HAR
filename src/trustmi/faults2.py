"""Two more everyday faults for the journal paper: a stuck sensor and saturation (Addition 1 of
docs/plan_additions_2026-11-14.md, approved and committed before any result of it).

Built ON trustmi.faults.FaultBank (which is NOT edited) and its physical units, channel roles and padding handling;
the same rules as every fault: only valid samples change, padding stays exactly 0, the valid length never changes.
Both faults are deterministic (no random draw), so the results are exactly reproducible.

  f_stuck   the sensor freezes and never recovers: from the onset to the last valid sample it repeats its value at
            the onset. The frozen stretch covers 25 / 50 / 100 % of the valid samples, so the onset is at 75 / 50 /
            0 % of the clip (100 % = the whole clip is one constant value, the first sample). The stretch always
            ends at the last valid sample; that fixes the onset (plan 1.2, wording correction of 14 Nov).
            Holding a value is the same in physical and normalised units.
  f_sat     readings clipped at the sensor's measuring range, the standard full-scale settings of MEMS motion
            sensors (plan 1.2, decision D2): level 1 accelerometer +/-8 g, gyroscope +/-1000 deg/s; level 2 +/-4 g,
            +/-500 deg/s; level 3 +/-2 g, +/-250 deg/s. Per axis, in physical units (units as trustmi.faults.UNITS).
            Orientation angles and the skeleton have no such range and get no saturation.
Video: stuck = the frame at the onset repeated to the end; saturation = over-exposure, pixel values x 1.5 / 2 / 3
clipped to the pixel range, made on the decoded frames by scripts/extract_video_faults2.py; VideoFaultBank2 looks
the resulting tokens up exactly as VideoFaultBank does.
"""
import numpy as np
from trustmi.faults import FaultBank, VideoFaultBank, G, ROOT, UNITS

FAULTS2 = ("f_stuck", "f_sat")
LEVELS = (1, 2, 3)
STUCK = {1: 0.25, 2: 0.50, 3: 1.00}                        # share of the valid samples that is frozen
SAT = {1: dict(acc_g=8.0, gyro_dps=1000.0), 2: dict(acc_g=4.0, gyro_dps=500.0), 3: dict(acc_g=2.0, gyro_dps=250.0)}
VIDEO_GAIN = {1: 1.5, 2: 2.0, 3: 3.0}                      # over-exposure: pixel value x gain, clipped at 255
APPLIES2 = {                                               # plan 1.3
    ("d1v2", "imu"): FAULTS2, ("d1v2", "skel"): ("f_stuck",), ("d1v2", "video"): FAULTS2,
    ("d2v", "watch"): FAULTS2, ("d2v", "phone"): FAULTS2, ("d2v", "video"): FAULTS2,
    ("d3", "arms"): FAULTS2, ("d3", "trunk"): FAULTS2, ("d3", "legs"): FAULTS2,
}
VIDEO_FAULTS2 = {"d1v2": ROOT / "data/features/video_r2plus1d18_v2_faults2.npz",
                 "d2v": ROOT / "data/features/mmact_video_r2plus1d18_v2_faults2.npz"}


def faults2_for(dataset, sensor):
    return APPLIES2.get((dataset, sensor), ())


def stuck_span(share, n):
    """(onset, frozen length) for n valid samples: the frozen stretch ends at the last valid sample."""
    g = min(max(1, int(round(share * int(n)))), int(n))
    return int(n) - g, g


class FaultBank2(FaultBank):
    """Stuck sensor and saturation for one (journal dataset, motion sensor or skeleton) on one evaluation split."""

    def __init__(self, dataset, modality, ds, train_ds, fold, seed=0):
        if not faults2_for(dataset, modality) or modality == "video":
            raise ValueError(f"no stuck / saturation faults planned for {dataset}/{modality} here")
        super().__init__(dataset, modality, ds, train_ds, fold, seed)

    def _stuck(self, i, level):
        n = int(self.lens[i])
        a, g = stuck_span(STUCK[level], n)
        out = np.zeros((self.T, self.C))
        out[:n] = self.x[i, :n]
        out[a:n] = self.x[i, a]
        return out, dict(stuck_start=a, stuck_len=g)

    def _sat(self, i, level):
        L, P, U = self.layout, SAT[level], UNITS[(self.base, self.modality)]
        if not (L.accel or L.gyro):
            raise ValueError(f"no measuring range for {self.dataset}/{self.modality}")
        p, n = self._phys(i)
        p = p.copy()
        before = p.copy()
        if L.accel:
            r = P["acc_g"] * (1.0 if U["accel"] == "g" else G)
            p[:, list(L.accel)] = np.clip(p[:, list(L.accel)], -r, r)
        if L.gyro:
            r = P["gyro_dps"] if U["gyro"] == "deg/s" else np.deg2rad(P["gyro_dps"])
            p[:, list(L.gyro)] = np.clip(p[:, list(L.gyro)], -r, r)
        ch = list(L.accel) + list(L.gyro)
        clipped = before != p                                     # (n, C); only accelerometer / gyroscope can clip
        out = np.zeros((self.T, self.C))
        out[:n] = np.where(clipped, self._back(i, p, n)[:n], self.x[i, :n])   # every unclipped value exactly as given
        return out, dict(clipped=int(clipped.sum()), n_values=int(n * len(ch)))

    def apply_fault(self, fault, level):
        """Every clip with this fault: (x_new float32 in the sensor's own shape, available (N,) all True, info)."""
        if fault not in faults2_for(self.dataset, self.modality):
            raise ValueError(f"{fault} is not planned for {self.dataset}/{self.modality}")
        fn = {"f_stuck": self._stuck, "f_sat": self._sat}[fault]
        x_new, infos = np.zeros_like(self.x), []
        for i in range(self.N):
            x_new[i], info = fn(i, level)
            infos.append(dict(clip=int(self.gidx[i]), fault=fault, level=level, **info))
        return x_new.reshape(self.shape).astype(np.float32), np.ones(self.N, bool), infos


class VideoFaultBank2(VideoFaultBank):
    """Frame-level stuck / over-exposure video features (scripts/extract_video_faults2.py), looked up and
    normalised exactly as VideoFaultBank does, with the same check of the stored clean tokens."""

    def __init__(self, dataset, ds, train_ds, fold, path=None):
        super().__init__(dataset, ds, train_ds, fold, path or VIDEO_FAULTS2[dataset])

    def apply_fault(self, fault, level):
        if fault not in faults2_for(self.dataset, "video"):
            raise ValueError(f"{fault} is not planned for {self.dataset}/video")
        x = self._norm(self._f[f"{fault}_L{level}"])
        return x, np.ones(self.N, bool), [dict(fault=fault, level=level)] * self.N
