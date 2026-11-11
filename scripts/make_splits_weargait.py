"""WearGait-PD plan 2.5: 10 person-wise folds, fixed once (seed 0), never changed.
Each fold: 10 Parkinson's people and 8 or 9 controls. Fold k: test = fold k, val = fold k+1 (cyclic),
train = the other 8 folds. Real-dropout clips (clean == False) never enter train / val; they are an extra test
set for the test people (plan 2.6).
    python scripts/make_splits_weargait.py
"""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
C = pd.read_parquet(ROOT / "data/processed/weargait_v1_clips.parquet")
K, SEED = 10, 0

rng = np.random.default_rng(SEED)
cohort = C.groupby("person").group.first()
members = {k: [] for k in range(K)}
for g in ("PD", "control"):
    people = np.array(sorted(cohort[cohort == g].index))
    for k, chunk in enumerate(np.array_split(rng.permutation(people), K)):
        members[k] += sorted(chunk.tolist())
order_hash = hashlib.sha256("\n".join(C.file + "#" + C.segment.astype(str) + "@" + C.start.astype(str))
                            .encode()).hexdigest()[:16]
folds = []
for k in range(K):
    v = (k + 1) % K
    folds.append(dict(fold=k, test=members[k], val=members[v],
                      train=sorted(p for j in range(K) if j not in (k, v) for p in members[j])))
out = ROOT / "splits/weargait_v1.json"
out.write_text(json.dumps(dict(dataset="weargait_v1", protocol=f"{K} person-wise folds, PD/control balanced",
                               seed=SEED, clip_filter="clean == True",
                               extra_test="clean == False (real dropouts), test people only",
                               clip_order_hash=order_hash, folds=folds), indent=1))

seen = pd.Series(0, index=cohort.index)
clean = C[C.clean]
for f in folds:
    sets = [set(f[p]) for p in ("test", "val", "train")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert set().union(*sets) == set(cohort.index)
    seen[f["test"]] += 1
    te, tr = clean[clean.person.isin(f["test"])], clean[clean.person.isin(f["train"])]
    nd = (~C.clean & C.person.isin(f["test"])).sum()
    print(f"fold {f['fold']}: test {len(f['test'])} people ({(cohort[f['test']] == 'PD').sum()} PD), "
          f"{len(te)} clean clips, classes test {te.y.nunique()} / train {tr.y.nunique()} | real-dropout test {nd}")
    assert te.y.nunique() == tr.y.nunique() == 6, "a fold misses a class"
assert (seen == 1).all()
print("every person is tested exactly once | clip order hash:", order_hash, "->", out.relative_to(ROOT))
