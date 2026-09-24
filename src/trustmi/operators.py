"""Counterfactual operator library (v1): interventions on ONE modality of a test sample.

Three families, plus a control:
  harmless  h_jitter, h_gain, h_rotate, h_angle, h_all   meaning-preserving, physically bounded
  semantic  s_swap        the modality is replaced by the same subject's recording of a DIFFERENT action
  control   c_swap_same   the modality is replaced by the same subject's recording of the SAME action
  removal   r_fill        the modality is set to the training mean (normalised 0) over its valid samples
            r_mask        the modality is left as is and flagged `absent`; the model must hide its tokens

Design rules
  * "Harmless" is defined by physics (sensor gain, noise and mounting tolerances), never by whether
    the model's prediction changes. The model's flip rate under harmless operators is a MEASUREMENT.
  * Everything is computed in physical units: normalised input -> physical (training-fold mean/std),
    perturb, clip to the physical range seen in TRAINING subjects, -> normalised.
  * Swaps keep everything except meaning fixed: same subject, same confound group (UTD-MHAD:
    wrist/thigh placement; MMAct: scene), and on MMAct the donor is resampled to the target's valid
    length, so the padding pattern cannot give the swap away. If no donor satisfies every rule,
    the swap is reported as unavailable; the rules are never relaxed silently.
  * Every random draw is keyed on (seed, operator, level, clip index), so results do not depend
    on batch order and are exactly reproducible.
  * Only padding-free (valid) samples are ever changed; padding stays exactly 0.
Severity levels are PROVISIONAL (blueprint: "Requires justification") and are swept, not tuned.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATS_DIR = ROOT / "data/processed/norm_stats"
MMACT_TENSORS = ROOT / "data/processed/mmact_sensor_v1.npz"

OPS = ("clean", "h_jitter", "h_gain", "h_rotate", "h_angle", "h_all",
       "s_swap", "c_swap_same", "r_fill", "r_mask")
FAMILY = {"clean": "clean", "s_swap": "semantic", "c_swap_same": "control",
          "r_fill": "removal", "r_mask": "removal",
          **{o: "harmless" for o in OPS if o.startswith("h_")}}

# jitter: noise SD as a fraction of the training SD of each channel; gain: +/- fraction per channel;
# rot_deg: max mounting rotation of the device (accelerometer and gyroscope rotate together);
# ang_deg: max constant offset added to each orientation angle.
LEVELS = {1: dict(jitter=0.02, gain=0.02, rot_deg=3.0, ang_deg=3.0),
          2: dict(jitter=0.05, gain=0.05, rot_deg=7.0, ang_deg=7.0),
          3: dict(jitter=0.10, gain=0.10, rot_deg=15.0, ang_deg=15.0)}


@dataclass(frozen=True)
class Layout:
    """Which channels of a modality tensor (T, C) play which physical role."""
    accel: tuple = ()
    gyro: tuple = ()
    ang_sin: tuple = ()      # sin of each orientation angle ...
    ang_cos: tuple = ()      # ... and the matching cos, same order
    generic: bool = False    # no physical model (skeleton, video features): swap/removal only


LAYOUTS = {
    ("d1", "imu"): Layout(accel=(0, 1, 2), gyro=(3, 4, 5)),                  # g, deg/s
    ("d2", "watch"): Layout(accel=(0, 1, 2)),                                 # m/s^2
    ("d2", "phone"): Layout(accel=(0, 1, 2), gyro=(3, 4, 5), ang_sin=(6, 7), ang_cos=(8, 9)),
}


def _stats_for(modality, fold, train_subjects):
    """Find the ONE saved normalisation file for this fold that holds `modality`, check its subjects."""
    hits = []
    for f in sorted(STATS_DIR.glob(f"*_fold{fold}.npz")):
        z = np.load(f)
        if f"{modality}_mean" in z.files:
            hits.append((f, z))
    if len(hits) != 1:
        raise RuntimeError(f"expected 1 stats file with '{modality}_mean' for fold {fold}, "
                           f"found {[str(h[0].name) for h in hits]}")
    f, z = hits[0]
    if "train_subjects" in z.files:
        saved = set(np.asarray(z["train_subjects"]).tolist())
        if saved != set(np.asarray(train_subjects).tolist()):
            raise RuntimeError(f"{f.name}: saved train subjects differ from the train dataset's")
    return z[f"{modality}_mean"].astype(np.float64), z[f"{modality}_std"].astype(np.float64), f.name


def _mmact_angle_unit():
    """Raw pitch/roll are columns 10:12 of the MMAct tensor. |values| > 2*pi means degrees."""
    x = np.load(MMACT_TENSORS)["X"][:, 10:12]
    return "deg" if np.nanmax(np.abs(x)) > 2 * np.pi + 0.1 else "rad"


def _rotation(rng, max_deg):
    """Rotation matrix about a uniformly random axis by an angle drawn from U(0, max_deg)."""
    k = rng.normal(size=3); k /= np.linalg.norm(k)
    a = np.deg2rad(rng.uniform(0.0, max_deg))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def _resample(x, n):
    """Linear resampling of x (L, C) to n samples along time."""
    L = len(x)
    if L == n:
        return x.copy()
    src, dst = np.linspace(0.0, 1.0, L), np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])], axis=1)


class OperatorBank:
    """Operators for one (dataset, modality) on one evaluation split.

    ds        the split the operators act on (normally the LOSO test subject)
    train_ds  the same fold's training split: used ONLY for the physical clipping range
    """

    def __init__(self, dataset, modality, ds, train_ds, fold, seed=0):
        self.dataset, self.modality, self.ds, self.fold, self.seed = dataset, modality, ds, fold, seed
        self.layout = LAYOUTS.get((dataset, modality), Layout(generic=True))
        x = np.asarray(ds.x[modality], dtype=np.float64)                # normalised input
        self.shape = x.shape                                            # e.g. skeleton (N, 40, 20, 3)
        self.x = x.reshape(x.shape[0], x.shape[1], -1)                  # always (N, T, C) inside the bank
        self.N, self.T, self.C = self.x.shape
        self.lens = (np.asarray(ds.x["len"]).astype(int) if "len" in ds.x
                     else np.full(self.N, self.T, int))
        self.y = np.asarray(ds.y).astype(int)
        self.subject = np.asarray(ds.subject)
        self.group = np.asarray(ds.group)
        self.gidx = np.asarray(getattr(ds, "idx", np.arange(self.N)))  # dataset-wide clip index
        self.angle_unit, self._train_ds = None, train_ds
        if self.layout.generic:
            return
        L = self.layout
        need = max(L.accel + L.gyro + L.ang_sin + L.ang_cos) + 1
        if self.C != need:
            raise RuntimeError(f"{dataset}/{modality}: expected (N, T, {need}) but got {self.x.shape}; "
                               "check the channel layout / axis order")
        tr_subjects = np.unique(np.asarray(train_ds.subject))
        if set(tr_subjects) & set(np.unique(self.subject)):
            raise RuntimeError("train_ds and ds share subjects: the clip range would leak")
        self.mu, self.sd, self.stats_file = _stats_for(modality, fold, tr_subjects)
        # physical range over VALID training samples, per channel
        xt = np.asarray(train_ds.x[modality], dtype=np.float64)
        lt = (np.asarray(train_ds.x["len"]).astype(int) if "len" in train_ds.x
              else np.full(len(xt), xt.shape[1], int))
        valid = np.arange(xt.shape[1])[None, :] < lt[:, None]
        phys = xt[valid] * self.sd + self.mu
        self.lo, self.hi = phys.min(0), phys.max(0)
        if self.layout.ang_sin:
            self.angle_unit = _mmact_angle_unit() if dataset == "d2" else "rad"

    # ---------------------------------------------------------------- helpers
    def _rng(self, op, level, i):
        return np.random.default_rng([self.seed, OPS.index(op), level, int(self.gidx[i])])

    def _phys(self, i):
        n = self.lens[i]
        return self.x[i, :n] * self.sd + self.mu, n

    def _back(self, i, p, n):
        out = np.zeros((self.T, self.C))
        out[:n] = (p - self.mu) / self.sd
        return out

    # ---------------------------------------------------------------- harmless
    def _harmless(self, i, parts, level):
        if self.layout.generic:
            raise ValueError(f"no physical model for modality '{self.modality}'")
        L, P = self.layout, LEVELS[level]
        rng = self._rng("h_all" if len(parts) > 1 else f"h_{parts[0]}", level, i)
        p, n = self._phys(i)
        p = p.copy()
        if "rotate" in parts:
            R = _rotation(rng, P["rot_deg"])
            for tri in (L.accel, L.gyro):
                if tri:
                    p[:, tri] = p[:, tri] @ R.T
        phys_ch = list(L.accel) + list(L.gyro)
        if "gain" in parts and phys_ch:
            p[:, phys_ch] *= rng.uniform(1 - P["gain"], 1 + P["gain"], size=len(phys_ch))
        if "jitter" in parts and phys_ch:
            p[:, phys_ch] += rng.normal(size=(n, len(phys_ch))) * P["jitter"] * self.sd[phys_ch]
        if "angle" in parts and L.ang_sin:
            s, c = list(L.ang_sin), list(L.ang_cos)
            ang = np.arctan2(p[:, s], p[:, c])
            d = rng.uniform(-P["ang_deg"], P["ang_deg"], size=len(s))
            ang = ang + (np.deg2rad(d) if self.angle_unit == "rad" else d)   # offset in argument units
            p[:, s], p[:, c] = np.sin(ang), np.cos(ang)
        clipped = 0
        if phys_ch:
            before = p[:, phys_ch].copy()
            p[:, phys_ch] = np.clip(p[:, phys_ch], self.lo[phys_ch], self.hi[phys_ch])
            clipped = int((before != p[:, phys_ch]).sum())
        return self._back(i, p, n), dict(clipped=clipped, n_values=n * max(len(phys_ch), 1))

    # ---------------------------------------------------------------- swaps
    def donors(self, i, same_label):
        ok = (self.subject == self.subject[i]) & (self.group == self.group[i])
        ok &= (self.y == self.y[i]) if same_label else (self.y != self.y[i])
        ok[i] = False
        return np.flatnonzero(ok)

    def _swap(self, i, op, same_label):
        cand = self.donors(i, same_label)
        if len(cand) == 0:
            return None, dict(available=False, donor=-1, donor_y=-1)
        j = int(self._rng(op, 0, i).choice(cand))
        out = np.zeros((self.T, self.C))
        out[:self.lens[i]] = _resample(self.x[j, :self.lens[j]], self.lens[i])
        return out, dict(available=True, donor=int(self.gidx[j]), donor_y=int(self.y[j]))

    # ---------------------------------------------------------------- public
    def apply_one(self, i, op, level=2):
        """Return (new modality tensor (T, C) normalised or None if unavailable, absent flag, info)."""
        if op == "clean":
            return self.x[i].copy(), False, {}
        if op.startswith("h_"):
            part = op[2:]
            if part == "angle" and not self.layout.ang_sin:
                return None, False, dict(available=False)      # modality has no orientation angles
            parts = ["rotate", "gain", "jitter", "angle"] if part == "all" else [part]
            out, info = self._harmless(i, parts, level)
            return out, False, info
        if op in ("s_swap", "c_swap_same"):
            out, info = self._swap(i, op, same_label=(op == "c_swap_same"))
            return out, False, info
        if op == "r_fill":
            return np.zeros((self.T, self.C)), False, {}
        if op == "r_mask":
            return self.x[i].copy(), True, {}
        raise ValueError(f"unknown operator {op!r}; choose from {OPS}")

    def apply(self, op, rows=None, level=2):
        """Apply `op` to every row. Returns (x_new (N, T, C) float32, absent (N,) bool,
        available (N,) bool, info list). Unavailable rows keep their clean input and are flagged."""
        rows = np.arange(self.N) if rows is None else np.asarray(rows)
        x_new = self.x[rows].copy()
        absent = np.zeros(len(rows), bool)
        avail = np.ones(len(rows), bool)
        infos = []
        for r, i in enumerate(rows):
            out, ab, info = self.apply_one(int(i), op, level)
            if out is None:
                avail[r] = False
            else:
                x_new[r] = out
            absent[r] = ab
            infos.append(dict(row=int(i), clip=int(self.gidx[i]), y=int(self.y[i]), **info))
        return x_new.reshape(len(rows), *self.shape[1:]).astype(np.float32), absent, avail, infos
