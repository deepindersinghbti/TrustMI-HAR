"""Sensor dropout (journal ablation): the mask builder and its effect on the fusion model, on FAKE data (CPU)."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trustmi.sensor_dropout import absent_mask, rate_of
from trustmi.models.fusion import FusionTransformer

MODS = ["watch", "phone", "video"]


def test_rate_is_read_from_the_model_name():
    assert rate_of("b4s3_sd50") == 0.5 and rate_of("b4_sd25") == 0.25
    assert rate_of("b4s3") == 0.0 and rate_of("b4s_nowatch") == 0.0 and rate_of("tmc_s3") == 0.0


def test_at_most_one_sensor_hidden_and_rate_respected():
    g = torch.Generator().manual_seed(0)
    m = absent_mask(20000, MODS, 0.5, g)
    stacked = torch.stack([m[s] for s in MODS])
    assert stacked.sum(0).max() == 1                                  # never two sensors of one clip
    assert abs(stacked.any(0).float().mean() - 0.5) < 0.02            # about half the clips lose one sensor
    share = stacked.float().sum(1) / stacked.sum()
    assert (share - 1 / 3).abs().max() < 0.02                         # each sensor equally often
    none = absent_mask(1000, MODS, 0.0, g)
    assert not torch.stack(list(none.values())).any()


def test_same_seed_same_mask():
    a = absent_mask(64, MODS, 0.5, torch.Generator().manual_seed(7))
    b = absent_mask(64, MODS, 0.5, torch.Generator().manual_seed(7))
    assert all(torch.equal(a[s], b[s]) for s in MODS)


def test_hidden_sensor_cannot_influence_the_answer():
    torch.manual_seed(0)
    for mods, x in ((MODS, dict(watch=torch.randn(6, 750, 3), phone=torch.randn(6, 750, 10),
                                video=torch.randn(6, 4, 512), len=torch.randint(200, 751, (6,)))),
                    (["video", "skel", "imu"], dict(video=torch.randn(6, 4, 512), skel=torch.randn(6, 40, 20, 3),
                                                    imu=torch.randn(6, 100, 6)))):
        model = FusionTransformer(mods, n_classes=5, d=32, layers=1).eval()
        mask = absent_mask(6, mods, 1.0, torch.Generator().manual_seed(1))    # every clip loses one sensor
        with torch.no_grad():
            a = model(x | {"absent": mask})
            y = dict(x)
            for s in mods:                                            # replace the hidden sensor by noise
                y[s] = torch.where(mask[s].view(-1, *[1] * (x[s].dim() - 1)), torch.randn_like(x[s]), x[s])
            b = model(y | {"absent": mask})
        assert torch.allclose(a, b, atol=1e-5)
        full = model(x)
        assert not torch.allclose(a, full, atol=1e-3)                 # and hiding does change something
