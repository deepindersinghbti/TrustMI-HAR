"""Plan step 4: per-clip MI scoring core on FAKE data (CPU). The cached, length-stretched reference bank must
give exactly what a slow direct computation gives."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from test_train_mi import fake_mmact
import cert_mi as CM
from trustmi.mi import MIMatcher, sensor_encoder, resample_to, pointwise_mi


def test_bank_cache_equals_direct_computation():
    torch.manual_seed(0)
    mods = ["watch", "phone"]
    model = MIMatcher({m: sensor_encoder(m, d=32, layers=1) for m in mods}, d=32, d_out=16).eval()
    tr, te = fake_mmact(20, 0), fake_mmact(6, 1)
    Xb = {m: tr.x[m][:5] for m in mods}; lb = tr.x["len"][:5]
    bank = CM.Bank(model, Xb, mods, lb)
    X = {m: te.x[m] for m in mods}; lens = te.x["len"]
    cols = CM.clip_mi(model, X, mods, bank, lens)
    for i in range(6):                                            # slow path, one clip at a time
        L = int(lens[i]); dst = torch.full((5,), L)
        zb = CM.embed_all(model, {m: resample_to(Xb[m], lb, dst) for m in mods}, mods, dst)
        zq = CM.embed_all(model, {m: X[m][i:i + 1] for m in mods}, mods, lens[i:i + 1])
        ref = pointwise_mi(zq["watch"], zq["phone"], zb["watch"], zb["phone"], model.scale()).item()
        assert abs(cols["pmi_watch_phone"][i] - ref) < 1e-5
    assert (cols["s4"] == -cols["mi_min"]).all()


def test_fixed_length_dataset_uses_one_shared_bank():
    torch.manual_seed(0)
    mods = ["imu", "skel"]
    model = MIMatcher({m: sensor_encoder(m, d=32, layers=1) for m in mods}, d=32, d_out=16).eval()
    Xb = {"imu": torch.randn(7, 100, 6), "skel": torch.randn(7, 40, 20, 3)}
    bank = CM.Bank(model, Xb, mods)
    X = {"imu": torch.randn(4, 100, 6), "skel": torch.randn(4, 40, 20, 3)}
    cols = CM.clip_mi(model, X, mods, bank)
    assert cols["pmi_imu_skel"].shape == (4,) and set(cols) >= {"s4", "mi_share_imu", "mi_share_skel"}
