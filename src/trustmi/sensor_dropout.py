"""Sensor dropout (also called modality dropout): a TRAINING-TIME ablation for the journal paper.

While training a fused model, each clip in a batch has, with probability `rate`, exactly ONE of its sensors
hidden from the fusion transformer (chosen uniformly among the model's sensors). Hiding uses the model's own
`absent` mask, i.e. the same mechanism as the r_mask operator and the hide-one-sensor sign: the hidden sensor's
tokens cannot be attended to. A clip never loses more than one sensor, so there is always something to learn from.
Validation (early stopping) and all tests use the complete clip, exactly as for the standard model.

Question it answers: does simply training with missing sensors make the fused model robust to a LYING sensor,
without any certificate? And can one such model replace the separate backup models for the "drop" action?

Rates are fixed in advance (2026-11-10, before any result) and are part of the model name, so a run ID fully
describes the run:  *_sd50 -> rate 0.50 (main ablation),  *_sd25 -> rate 0.25 (sensitivity).
"""
import torch

RATES = {"sd50": 0.50, "sd25": 0.25}
SEED_OFFSET = 5000          # the dropout draws use their own generator: the batch order stays that of train.py


def rate_of(model_name):
    """Sensor-dropout rate encoded in a model name (b4_sd50 -> 0.5); 0.0 for every other model."""
    return RATES.get(model_name.rsplit("_", 1)[-1], 0.0)


def absent_mask(batch_size, mods, rate, gen):
    """{sensor: (B,) bool}, True = hide this sensor for this clip. At most one True per clip."""
    hide = torch.rand(batch_size, generator=gen) < rate
    which = torch.randint(0, len(mods), (batch_size,), generator=gen)
    return {m: hide & (which == i) for i, m in enumerate(mods)}
