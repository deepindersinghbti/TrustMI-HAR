"""Check the operator library on real data (no model involved): D1 imu, D2 watch and phone, one fold.

Every check is an assert; the table shows how strong each operator is. Nothing is written to disk.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustmi.operators import OperatorBank, OPS, LEVELS, FAMILY


def valid_mask(bank):
    return np.arange(bank.T)[None, :] < bank.lens[:, None]


def check_bank(bank, name):
    x, V = bank.x, valid_mask(bank)
    print(f"\n=== {name}: {bank.N} clips, T={bank.T}, C={bank.C}"
          + ("" if bank.layout.generic else f" | stats {bank.stats_file}")
          + (f" | angle unit {bank.angle_unit}" if bank.angle_unit else ""))
    if not bank.layout.generic:
        L = bank.layout
        if L.ang_sin:
            p = x[V] * bank.sd + bank.mu
            r = p[:, list(L.ang_sin)] ** 2 + p[:, list(L.ang_cos)] ** 2
            print(f"    clean sin^2+cos^2: {r.min():.4f}-{r.max():.4f} (should be ~1)")
    rows = []
    for op in OPS:
        for level in (sorted(LEVELS) if op.startswith("h_") else [0]):
            if bank.layout.generic and FAMILY[op] == "harmless":
                continue
            xn, absent, avail, info = bank.apply(op, level=level or 2)
            assert xn.shape == bank.shape, op
            xn = xn.reshape(x.shape)                     # compare in the bank's (N, T, C) view
            assert np.all(xn[~V] == 0), f"{op}: padding changed"
            xn2, _, avail2, _ = bank.apply(op, level=level or 2)
            xn2 = xn2.reshape(x.shape)
            assert np.array_equal(xn, xn2) and np.array_equal(avail, avail2), f"{op}: not deterministic"
            if not avail.any():
                rows.append((op, level, "n/a", "", "", "")); continue
            a = avail
            d = (xn[a].astype(np.float64) - x[a])[V[a]]
            rms = float(np.sqrt((d ** 2).mean()))
            clip = sum(i.get("clipped", 0) for i in info) / max(1, sum(i.get("n_values", 0) for i in info))
            if op == "clean":
                assert rms == 0.0
            if op == "r_fill":
                assert np.all(xn == 0)
            if op == "r_mask":
                assert absent.all() and rms == 0.0
            else:
                assert not absent.any()
            if op == "h_rotate" and bank.layout.accel:
                # rotation must keep |accel| per sample; skip samples where clipping touched an axis
                ac = list(bank.layout.accel)
                p0 = (x * bank.sd + bank.mu)[..., ac]
                p1 = (xn.astype(np.float64) * bank.sd + bank.mu)[..., ac]
                edge = (np.isclose(p1, bank.lo[ac], atol=1e-4) | np.isclose(p1, bank.hi[ac], atol=1e-4)).any(-1)
                keep = V & ~edge
                m0, m1 = np.linalg.norm(p0, axis=-1)[keep], np.linalg.norm(p1, axis=-1)[keep]
                rel = np.abs(m1 - m0) / np.maximum(m0, 1e-6)
                assert keep.sum() > 0 and np.median(rel) < 1e-4, \
                    f"rotation changed |accel| (median rel {np.median(rel):.2e}, checked {keep.sum() / V.sum():.0%})"
            if op in ("s_swap", "c_swap_same"):
                for k, i in enumerate(info):
                    if not avail[k]:
                        continue
                    j = np.flatnonzero(bank.gidx == i["donor"])[0]
                    assert bank.subject[j] == bank.subject[k], "donor from another subject"
                    assert bank.group[j] == bank.group[k], "donor from another placement/scene group"
                    assert j != k, "donor is the clip itself"
                    assert (i["donor_y"] == i["y"]) == (op == "c_swap_same"), "donor label rule broken"
                    assert bank.y[j] == i["donor_y"]
            rows.append((op, level or "", f"{a.mean():.1%}", f"{rms:.3f}", f"{clip:.2%}", ""))
    print(f"    {'operator':<12}{'level':>6}{'avail':>8}{'RMS change (train SD)':>24}{'clipped':>10}")
    for op, lv, av, rms, cl, _ in rows:
        print(f"    {op:<12}{str(lv):>6}{av:>8}{rms:>24}{cl:>10}")
    if not bank.layout.generic:
        other = OperatorBank(bank.dataset, bank.modality, bank.ds, bank._train_ds, bank.fold, seed=bank.seed + 1)
        assert not np.array_equal(bank.apply("h_all")[0], other.apply("h_all")[0]), "seed has no effect"
    print("    all checks passed")


def main(fold=0):
    from trustmi.data.utd import UTDMHAD
    from trustmi.data.mmact import MMAct
    tr, te = UTDMHAD(fold, "train"), UTDMHAD(fold, "test")
    check_bank(OperatorBank("d1", "imu", te, tr, fold), f"D1 UTD-MHAD imu, fold {fold}")
    for m in ("skel", "video"):                          # generic sensors: swaps and removal only
        check_bank(OperatorBank("d1", m, te, tr, fold), f"D1 UTD-MHAD {m}, fold {fold}")
    tr, te = MMAct(fold, "train"), MMAct(fold, "test")
    for m in ("watch", "phone"):
        check_bank(OperatorBank("d2", m, te, tr, fold), f"D2 MMAct {m}, fold {fold}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
