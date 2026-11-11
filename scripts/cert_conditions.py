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


def parts_for(kind):
    """Which parts a condition set runs on: the no-watch bench needs no validation part (nothing is fitted on
    it) and adds the 'nowatch' part."""
    return ("test", "nowatch") if kind == "nowatch" else ("val", "test")


def dataset_for(kind, DS, dname, fold, part):
    """DS(fold, part), or the no-watch clips of the fold's test person for part 'nowatch'."""
    if part == "nowatch":
        if kind != "nowatch" or dname not in NOWATCH_DATASETS:
            raise ValueError("part 'nowatch' needs --conditions nowatch on d2v")
        from trustmi.data.mmact import MMActNoWatch
        return MMActNoWatch(fold)
    return DS(fold, part)


def folds_for(kind, dname, folds):
    """For the no-watch bench, keep only the folds whose test person has no-watch clips."""
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
    kind 'nowatch' -> the no-watch bench."""
    if kind == "lies":
        return build_conditions(dname, ds, tr, fold, attack, phys_sensors)
    if kind == "faults":
        return build_fault_conditions(dname, ds, tr, fold, video)
    if kind == "nowatch":
        return build_nowatch_conditions(dname, ds, tr, fold)
    raise ValueError(f"unknown condition set {kind!r}")


def conditions_suffix(kind, video=True):
    """File-name suffix of a condition set: '' for lies (unchanged names), '_faults' or '_faults-novideo',
    '_nowatch'."""
    if kind == "nowatch":
        return "_nowatch"
    return "" if kind == "lies" else ("_faults" if video else "_faults-novideo")
