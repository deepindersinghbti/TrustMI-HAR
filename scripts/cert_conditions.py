"""The test-bench conditions for one fold and part, built in ONE place (plan step 2).

Moved unchanged out of cert_experts.py so that the MI scripts use exactly the same inputs:
  clean                 the unchanged input
  s_swap:<sensor>       that sensor replaced by the same person's DIFFERENT action (the lie)
  c_swap_same:<sensor>  replaced by the same person's SAME action (control)
  h_all:<sensor>        harmless level-2 change (control; only sensors with a physical model)
The operators are deterministic, so calling this twice gives identical inputs.
"""
import numpy as np
from trustmi.operators import OperatorBank


def build_conditions(dname, ds, tr, fold, attack, phys_sensors):
    """Returns (conds, gidx) where conds[name] = (replacements {sensor: array}, donor_y, available, attacked)
    and gidx are the clip ids of ds. attack: sensors to plant lies in; phys_sensors: sensors that get the
    harmless condition (CFG[d]["sensors"])."""
    N = len(ds.y)
    banks = {sen: OperatorBank(dname, sen, ds, tr, fold) for sen in attack}
    gidx = banks[attack[0]].gidx
    conds = {"clean": ({}, np.full(N, -1), np.ones(N, bool), "")}
    for sen in attack:
        for op in ("s_swap", "c_swap_same"):
            xn, _, avail, info = banks[sen].apply(op)
            conds[f"{op}:{sen}"] = ({sen: xn}, np.array([i["donor_y"] for i in info]), avail, sen)
        if sen in phys_sensors:                  # harmless changes need a physical model
            xn, _, avail, _ = banks[sen].apply("h_all", level=2)
            conds[f"h_all:{sen}"] = ({sen: xn}, np.full(N, -1), avail, sen)
    return conds, gidx


# ---------------------------------------------------------------- journal: real sensor faults
# docs/plan_faults_2026-11-11.md, step 2. A separate function, so build_conditions above stays exactly as it was.
def build_fault_conditions(dname, ds, tr, fold, video=True):
    """The fault test bench for one fold and part, in the same format as build_conditions:
      clean                       the unchanged input
      <fault>:L<level>:<sensor>   that sensor with fault f_drift / f_gap / f_shift at level 1-3
                                  (only where the plan allows it, trustmi.faults.APPLIES)
    donor_y is -1 (a fault has no donor action); `attacked` is the faulty sensor. Faults are deterministic,
    so calling this twice gives identical inputs. video=False leaves the video faults out (only for a smoke
    run made before the frame-level video fault features exist; the output file name then says so)."""
    from trustmi.faults import FaultBank, VideoFaultBank, APPLIES, LEVELS, condition_name
    N = len(ds.y)
    sensors = [s for (d, s) in APPLIES if d == dname and (video or s != "video")]
    if not sensors:
        raise ValueError(f"no faults planned for {dname}")
    banks = {s: (VideoFaultBank(dname, ds, tr, fold) if s == "video" else FaultBank(dname, s, ds, tr, fold))
             for s in sensors}
    gidx = next(b.gidx for s, b in banks.items() if s != "video")
    conds = {"clean": ({}, np.full(N, -1), np.ones(N, bool), "")}
    for s, bank in banks.items():
        for fault in APPLIES[(dname, s)]:
            for lv in LEVELS:
                xn, avail, _ = bank.apply_fault(fault, lv)
                conds[condition_name(fault, lv, s)] = ({s: xn}, np.full(N, -1), avail, s)
    return conds, gidx


# ---------------------------------------------------------------- journal: the clips without a watch
# docs/plan_nowatch_2026-11-11.md. Separate functions, so everything above stays exactly as it was.
NOWATCH_DATASETS = ("d2v",)


def build_nowatch_conditions(dname, ds, tr, fold):
    """The no-watch bench for one fold, in the same format as build_conditions:
      part 'test'     clean          the test person's normal clips, unchanged (checked against the stored run)
                      r_fill:watch   the same clips with the watch set to its training mean (the r_fill operator)
      part 'nowatch'  nw_fill:watch  the clips where the watch was never recorded; the loader already sets the
                                     watch to the training mean (= the r_fill rule), so nothing is replaced
    donor_y is -1; `attacked` is the watch. Deterministic."""
    if dname not in NOWATCH_DATASETS:
        raise ValueError(f"no no-watch clips in {dname}")
    N = len(ds.y)
    if getattr(ds, "nowatch", False):
        return {"nw_fill:watch": ({}, np.full(N, -1), np.ones(N, bool), "watch")}, np.asarray(ds.idx)
    bank = OperatorBank(dname, "watch", ds, tr, fold)
    xn, _, avail, _ = bank.apply("r_fill")
    conds = {"clean": ({}, np.full(N, -1), np.ones(N, bool), ""),
             "r_fill:watch": ({"watch": xn}, np.full(N, -1), avail, "watch")}
    return conds, bank.gidx


# ---------------------------------------------------------------- journal: WearGait-PD real dropouts
# docs/plan_weargait_2026-11-11.md, section 2.6. Separate function; everything above stays as it was.
REALDROP_DATASETS = ("d3",)


def build_realdrop_conditions(dname, ds, tr, fold):
    """The real-dropout bench for one fold, in the same format as build_conditions:
      part 'test'     clean            the test people's clean clips, unchanged (checked against the stored run)
                      r_fill:<sensor>  the same clips with that sensor group set to its training mean (r_fill)
      part 'dropout'  rd_fill:<groups> the clips where those sensor groups REALLY lost samples; the loader has
                                       already set the missing samples to the training mean, so nothing is
                                       replaced. One condition per pattern of missing groups (e.g. 'legs',
                                       'arms,trunk'); a clip is available only under its own pattern.
    donor_y is -1; `attacked` is the group(s) with missing data. Deterministic."""
    if dname not in REALDROP_DATASETS:
        raise ValueError(f"no real-dropout clips in {dname}")
    N = len(ds.y)
    if getattr(ds, "part", None) == "dropout":
        miss = np.asarray(ds.missing)
        conds = {f"rd_fill:{pat}": ({}, np.full(N, -1), miss == pat, pat) for pat in sorted(set(miss))}
        return conds, np.asarray(ds.idx)
    conds = {"clean": ({}, np.full(N, -1), np.ones(N, bool), "")}
    gidx = None
    for s in ("arms", "trunk", "legs"):
        bank = OperatorBank(dname, s, ds, tr, fold)
        xn, _, avail, _ = bank.apply("r_fill")
        conds[f"r_fill:{s}"] = ({s: xn}, np.full(N, -1), avail, s)
        gidx = bank.gidx
    return conds, gidx


# ---------------------------------------------------------------- journal: WearGait-PD IMUs switched off
# docs/plan_followup_2026-11-12.md, Part A. Separate functions; everything above stays as it was.
IMUOFF_DATASETS = ("d3",)
IMUS = {"arms": 2, "trunk": 3, "legs": 8}     # IMUs per group, 6 channels each, in scripts/build_weargait.py order
IMUOFF_SEED = 0


def imu_order(group, clip):
    """The random order in which this clip's IMUs of `group` are switched off. Keyed on (group, clip id) only,
    so it is the same for every seed and every model."""
    rng = np.random.default_rng([IMUOFF_SEED, list(IMUS).index(group), int(clip)])
    return rng.permutation(IMUS[group])


def imus_off(group, gidx, k):
    """Boolean (N, n_imus): True where the IMU is among the first k of the clip's order (nested in k)."""
    order = np.stack([imu_order(group, c) for c in gidx]) if len(gidx) else np.zeros((0, IMUS[group]), int)
    rank = np.argsort(order, axis=1)          # rank[r, j] = position of IMU j in clip r's order
    return rank < k


def imuoff_input(x, group, gidx, k):
    """x: the normalised group tensor (N, T, 6 * n_imus). Returns a float32 copy in which all 6 channels of the
    first k IMUs of each clip's order are 0 (= the training mean) for the whole clip; everything else untouched."""
    x = np.asarray(x, dtype=np.float32)
    n = IMUS[group]
    if x.ndim != 3 or x.shape[2] != 6 * n:
        raise RuntimeError(f"{group}: expected (N, T, {6 * n}), got {x.shape}")
    off = np.repeat(imus_off(group, gidx, k), 6, axis=1)          # (N, 6n) channel mask
    return np.where(off[:, None, :], np.float32(0.0), x).astype(np.float32)


def build_imuoff_conditions(dname, ds, tr, fold):
    """The IMU-off bench for one fold (test part only), in the same format as build_conditions:
      clean               the test people's clean clips, unchanged (checked against the stored run)
      io_fill:<group>:k<k>  the first k IMUs of the clip's random order in that group set to the training mean
                          for the whole clip (k = 1 .. all; k = all is identical to r_fill:<group>)
    donor_y is -1; `attacked` is the group. Deterministic. tr and fold are unused (kept for the common signature):
    switching off needs no training statistics, because the training mean is 0 after normalisation."""
    if dname not in IMUOFF_DATASETS:
        raise ValueError(f"no IMU groups in {dname}")
    N = len(ds.y)
    gidx = np.asarray(ds.idx)
    conds = {"clean": ({}, np.full(N, -1), np.ones(N, bool), "")}
    for g, n in IMUS.items():
        x = np.asarray(ds.x[g])
        for k in range(1, n + 1):
            conds[f"io_fill:{g}:k{k}"] = ({g: imuoff_input(x, g, gidx, k)}, np.full(N, -1), np.ones(N, bool), g)
    return conds, gidx


def parts_for(kind):
    """Which parts a condition set runs on: the no-watch and real-dropout benches need no validation part
    (nothing is fitted on them) and add their own extra part; the IMU-off bench runs on the test part only."""
    return {"nowatch": ("test", "nowatch"), "realdrop": ("test", "dropout"),
            "imuoff": ("test",)}.get(kind, ("val", "test"))


def dataset_for(kind, DS, dname, fold, part):
    """DS(fold, part); the no-watch clips (part 'nowatch'); the real-dropout clips (part 'dropout')."""
    if part == "nowatch":
        if kind != "nowatch" or dname not in NOWATCH_DATASETS:
            raise ValueError("part 'nowatch' needs --conditions nowatch on d2v")
        from trustmi.data.mmact import MMActNoWatch
        return MMActNoWatch(fold)
    if part == "dropout" and (kind != "realdrop" or dname not in REALDROP_DATASETS):
        raise ValueError("part 'dropout' needs --conditions realdrop on d3")
    return DS(fold, part)


def folds_for(kind, dname, folds):
    """For the no-watch bench, keep only the folds whose test person has no-watch clips."""
    if kind == "realdrop":
        if dname not in REALDROP_DATASETS:
            raise SystemExit(f"--conditions realdrop: no real-dropout clips in {dname}")
        return folds
    if kind == "imuoff":
        if dname not in IMUOFF_DATASETS:
            raise SystemExit(f"--conditions imuoff: no IMU groups in {dname}")
        return folds
    if kind != "nowatch":
        return folds
    if dname not in NOWATCH_DATASETS:
        raise SystemExit(f"--conditions nowatch: no no-watch clips in {dname}")
    from trustmi.data.mmact import nowatch_folds
    keep = [f for f in folds if f in set(nowatch_folds())]
    if not keep:
        raise SystemExit(f"--conditions nowatch: none of folds {folds} has no-watch clips")
    return keep


def conditions_for(kind, dname, ds, tr, fold, attack, phys_sensors, video=True):
    """kind 'lies' -> build_conditions (the default everywhere, unchanged); kind 'faults' -> the fault bench;
    kind 'nowatch' -> the no-watch bench; 'realdrop' -> the real-dropout bench; 'imuoff' -> the IMU-off bench."""
    if kind == "lies":
        return build_conditions(dname, ds, tr, fold, attack, phys_sensors)
    if kind == "faults":
        return build_fault_conditions(dname, ds, tr, fold, video)
    if kind == "nowatch":
        return build_nowatch_conditions(dname, ds, tr, fold)
    if kind == "realdrop":
        return build_realdrop_conditions(dname, ds, tr, fold)
    if kind == "imuoff":
        return build_imuoff_conditions(dname, ds, tr, fold)
    raise ValueError(f"unknown condition set {kind!r}")


def conditions_suffix(kind, video=True):
    """File-name suffix of a condition set: '' for lies (unchanged names), '_faults' or '_faults-novideo',
    '_nowatch', '_realdrop', '_imuoff'."""
    if kind in ("nowatch", "realdrop", "imuoff"):
        return f"_{kind}"
    return "" if kind == "lies" else ("_faults" if video else "_faults-novideo")
