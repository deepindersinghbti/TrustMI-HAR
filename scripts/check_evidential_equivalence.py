"""Check that our TMC / ECML re-implementation (src/trustmi/models/evidential.py) is numerically identical
to the official code, on random evidence. The official repositories are cloned OUTSIDE this project
(they carry no licence, so none of their code is copied in); pass their paths:
  python scripts/check_evidential_equivalence.py --rcml ~/ext/RCML --tmc "~/ext/TMC/TMC ICLR"
"""
import argparse, importlib.util, sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trustmi.models.evidential import fuse_tmc, fuse_ecml, method_loss


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rcml", required=True); ap.add_argument("--tmc", required=True)
    a = ap.parse_args()
    rc_loss = load("rcml_loss", Path(a.rcml).expanduser() / "loss_function.py")
    rc_model = load("rcml_model", Path(a.rcml).expanduser() / "model.py")
    tmc_model = load("tmc_model", Path(a.tmc).expanduser() / "model.py")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    worst = 0.0
    for V, K in ((2, 35), (3, 27)):
        for epoch in (1, 3, 10):
            anneal = 6
            ev = [nn.functional.softplus(3 * torch.randn(64, K, device=dev)) for _ in range(V)]
            y = torch.randint(0, K, (64,), device=dev)
            coef = min(1.0, epoch / anneal)
            # ECML: their model's fusion (evidence collectors replaced by identity) and their loss
            rm = rc_model.RCML(V, [[K]] * V, K).to(dev)
            rm.EvidenceCollectors = nn.ModuleList([nn.Identity() for _ in range(V)])
            their_ev, their_ea = rm([e for e in ev])
            their = rc_loss.get_loss(their_ev, their_ea, y, epoch, K, anneal, 1.0, dev)
            ours = method_loss("ecml", ev, y, coef, 1.0)
            d_f = (fuse_ecml(ev) - (their_ea + 1)).abs().max().item()
            d_l = abs(ours.item() - their.item()) / abs(their.item())
            # TMC: their Dempster combination and their forward loss (classifiers replaced by identity)
            tm = tmc_model.TMC(K, V, [[K]] * V, lambda_epochs=anneal).to(dev)
            tm.Classifiers = nn.ModuleList([nn.Identity() for _ in range(V)])
            _, their_ea_t, their_t = tm({v: ev[v] for v in range(V)}, y, epoch)
            ours_t = method_loss("tmc", ev, y, coef)
            d_ft = (fuse_tmc([e + 1 for e in ev]) - (their_ea_t + 1)).abs().max().item()
            d_lt = abs(ours_t.item() - their_t.item()) / abs(their_t.item())
            worst = max(worst, d_f, d_l, d_ft, d_lt)
            print(f"V={V} K={K} epoch={epoch:2d} | ECML fusion {d_f:.2e} loss(rel) {d_l:.2e} | "
                  f"TMC fusion {d_ft:.2e} loss(rel) {d_lt:.2e}")
    print("EQUIVALENT" if worst < 1e-4 else f"MISMATCH (worst {worst:.2e}): stop and paste this")


if __name__ == "__main__":
    main()
