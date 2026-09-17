import sys, numpy as np, torch
sys.path.insert(0, "src")
from trustmi.data.utd import UTDMHAD, _load_all, fit_stats

data, split = _load_all()
for fold in range(8):
    tr, va, te = (UTDMHAD(fold, p) for p in ("train", "val", "test"))
    subs = [set(d.subject.tolist()) for d in (tr, va, te)]
    assert not (subs[0] & subs[1]) and not (subs[0] & subs[2]) and not (subs[1] & subs[2])
    assert len(tr) + len(va) + len(te) == 861

    # Training data must be standardised; test data only approximately.
    m = tr.x["imu"].double().mean(dim=(0, 1)).abs().max().item()
    s = tr.x["imu"].double().std(dim=(0, 1)).sub(1).abs().max().item()
    assert m < 1e-4 and s < 1e-2, (fold, m, s)

    # Leakage check: statistics must change if the test subject is swapped into training.
    leaky = fit_stats(data, np.concatenate([tr.idx, te.idx]))
    clean = fit_stats(data, tr.idx)
    assert not np.allclose(leaky["imu"][0], clean["imu"][0]), "stats ignore train_idx?"

    if fold == 0:
        b = tr[0]
        print({k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in b.items()})
    te_m = te.x["imu"].double().mean(dim=(0, 1)).abs().max().item()
    print(f"fold {fold}: train {len(tr)} val {len(va)} test {len(te)} | "
          f"test-set IMU |mean| max {te_m:.3f}")
print("all loader checks passed")
