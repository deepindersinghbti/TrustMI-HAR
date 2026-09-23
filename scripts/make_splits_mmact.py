"""LOSO splits for MMAct (20 folds): test = subject k, val = next subject (cyclic), train = the rest.
Only clips with status == 'main' are used; 'no_watch' clips of the test subject form a separate test set."""
import hashlib
import json
import numpy as np

d = np.load("data/processed/mmact_sensor_v1.npz")
subj, status, trial, y = d["subject"], d["status"], d["trial"], d["y"]
subjects = sorted(int(s) for s in np.unique(subj))
order_hash = hashlib.sha256("\n".join(trial).encode()).hexdigest()[:16]

folds = []
for k, s in enumerate(subjects):
    v = subjects[(k + 1) % len(subjects)]
    folds.append(dict(fold=k, test=[s], val=[v], train=[x for x in subjects if x not in (s, v)]))
with open("splits/mmact_loso_v1.json", "w") as f:
    json.dump(dict(dataset="mmact_sensor_v1", protocol="loso", clip_filter="status == main",
                   extra_test="status == no_watch, test subject only",
                   trial_order_hash=order_hash, folds=folds), f, indent=1)

main = status == "main"
seen = np.zeros(len(subj), int)
for f in folds:
    te = main & np.isin(subj, f["test"])
    va = main & np.isin(subj, f["val"])
    tr = main & np.isin(subj, f["train"])
    assert not (te & va).any() and not (te & tr).any() and not (va & tr).any()
    assert te.sum() + va.sum() + tr.sum() == main.sum()
    seen += te
    nw = (status == "no_watch") & np.isin(subj, f["test"])
    print(f"fold {f['fold']:2d}: test s{f['test'][0]:2d} ({te.sum():3d})  val s{f['val'][0]:2d} ({va.sum():3d})"
          f"  train {tr.sum()} | classes: test {len(np.unique(y[te]))} train {len(np.unique(y[tr]))}"
          f" | no_watch test {nw.sum()}")
assert (seen[main] == 1).all()
print("every main clip is tested exactly once | trial order hash:", order_hash)
