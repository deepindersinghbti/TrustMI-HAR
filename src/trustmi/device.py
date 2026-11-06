"""Where the scoring scripts run: the GPU when there is one, otherwise the CPU (e.g. the Windows laptop).

  TRUSTMI_DEVICE=auto (default) | cuda | cpu      choose the device
  TRUSTMI_AUTOCAST=1 (default) | 0                bf16 autocast on/off

On the GPU this is exactly the previous behaviour: model on "cuda", torch.autocast("cuda", bfloat16).
On the CPU the same bf16 autocast is used, so outputs stay close to the stored GPU predictions;
every script still checks clean outputs against preds_*.parquet before using them.
Training scripts (train.py, train_mi.py) are unchanged and still need the GPU.
"""
import os
import torch


def get_device():
    want = os.environ.get("TRUSTMI_DEVICE", "auto").strip().lower()
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want not in ("cuda", "cpu"):
        raise SystemExit(f"TRUSTMI_DEVICE must be auto, cuda or cpu, not {want!r}")
    if want == "cuda" and not torch.cuda.is_available():
        raise SystemExit("TRUSTMI_DEVICE=cuda but no CUDA GPU is available")
    return torch.device(want)


DEVICE = get_device()
AUTOCAST = os.environ.get("TRUSTMI_AUTOCAST", "1").strip() != "0"


def autocast():
    """bf16 autocast on DEVICE (a no-op context when TRUSTMI_AUTOCAST=0)."""
    return torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=AUTOCAST)
