import json, hashlib
import numpy as np

meta = np.load("data/processed/utd_mhad_v1.npz", allow_pickle=True)
subj, trials = meta["subject"], meta["trial"]
subjects = sorted(set(subj.tolist()))

folds = []
for k, test_s in enumerate(subjects):
    val_s = subjects[(k + 1) % len(subjects)]
    train_s = [s for s in subjects if s not in (test_s, val_s)]
    idx = {name: np.where(np.isin(subj, ss))[0].tolist()
           for name, ss in [("train", train_s), ("val", [val_s]), ("test", [test_s])]}
    # N7 integrity checks
    assert not set(idx["train"]) & set(idx["val"])
    assert not set(idx["train"]) & set(idx["test"])
    assert not set(idx["val"]) & set(idx["test"])
    assert len(idx["train"]) + len(idx["val"]) + len(idx["test"]) == len(subj)
    folds.append(dict(fold=k, test_subject=test_s, val_subject=val_s,
                      train_subjects=train_s, **idx))

trial_hash = hashlib.sha256("|".join(trials.tolist()).encode()).hexdigest()[:16]
out = dict(dataset="UTD-MHAD", version="v1", scheme="LOSO + rotating inner-val subject",
           n_trials=len(subj), trial_order_sha=trial_hash, folds=folds)
with open("splits/utd_mhad_loso_v1.json", "w") as f:
    json.dump(out, f, indent=1)

for fd in folds:
    print(f"fold {fd['fold']}: test s{fd['test_subject']} ({len(fd['test'])})  "
          f"val s{fd['val_subject']} ({len(fd['val'])})  train {len(fd['train'])}")
print("trial order hash:", trial_hash)
