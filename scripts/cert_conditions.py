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
