"""Summarise MMAct (d2) sensor baselines: accuracy, fusion gain, pooled errors, and what the two
single-device models said when B4-sensor was wrong (the 'agree but wrong' cases behind H1)."""
import glob
import numpy as np
import pandas as pd

cls = np.load("data/processed/mmact_sensor_v1.npz")["classes"]
df = pd.concat(pd.read_parquet(f) for f in glob.glob("results/raw/*_d2_*.parquet"))
t = df[(df.part == "test") & (df.condition == "clean")]
base = t[t.block == "BASE"]

per_fold = base.groupby(["model", "metric", "fold"]).value.mean().reset_index()
s = per_fold.groupby(["model", "metric"]).value.agg(["mean", "std", "min"]).unstack("metric").round(3)
print("1) BASELINES (mean over 20 folds of seed-averaged test scores)")
print(s[[("mean", "acc"), ("std", "acc"), ("min", "acc"), ("mean", "macro_f1"), ("std", "macro_f1")]].to_string())
print("runs per model:", base[base.metric == "acc"].groupby("model").size().to_dict())
sd = base[base.metric == "acc"].groupby(["model", "fold"]).value.std().groupby("model").mean()
print("mean seed-to-seed SD of accuracy:", sd.round(3).to_dict())

acc = per_fold[per_fold.metric == "acc"].pivot(index="fold", columns="model", values="value")
acc["gain"] = acc.b4s - acc[["b1_watch", "b1_phone"]].max(axis=1)
print("\n2) PER-SUBJECT TEST ACCURACY (seed mean; fold k = subject k+1; gain = b4s - best single)")
print(acc.round(3).to_string())
print(f"b4s beats the best single device in {(acc.gain > 0).sum()}/20 folds | mean gain {acc.gain.mean():+.3f}")

P = pd.concat(pd.read_parquet(f, columns=["exp_id", "trial", "group", "true", "pred", "msp"])
              for f in glob.glob("runs/BASE_d2_*/preds_test.parquet"))
P["model"] = P.exp_id.str.extract(r"BASE_d2_(.+)_loso")[0]
P["seed"] = P.exp_id.str.extract(r"_s(\d+)$")[0].astype(int)
P["err"] = P.true != P.pred
e = P.groupby("model").err.agg(n="size", errors="sum", error_rate="mean")
print("\n3) POOLED TEST ERRORS (all folds, all seeds)")
print(e.round(4).to_string())

w = P.pivot_table(index=["seed", "trial"], columns="model", values="pred", aggfunc="first")
w = w.join(P.drop_duplicates(["seed", "trial"]).set_index(["seed", "trial"]).true)
x = w[w.b4s != w.true]
agree = x.b1_watch == x.b1_phone
cats = {
    "watch & phone both gave B4's wrong answer": agree & (x.b1_watch == x.b4s),
    "watch & phone agreed on another wrong answer": agree & (x.b1_watch != x.b4s) & (x.b1_watch != x.true),
    "watch & phone both right (fusion broke it)": agree & (x.b1_watch == x.true),
    "watch & phone disagreed, one of them right": ~agree & ((x.b1_watch == x.true) | (x.b1_phone == x.true)),
    "watch & phone disagreed, both wrong": ~agree & (x.b1_watch != x.true) & (x.b1_phone != x.true),
}
print(f"\n4) WHEN B4-SENSOR WAS WRONG ({len(x)} errors), THE SINGLE-DEVICE MODELS...")
for k, m in cats.items():
    print(f"   {k:48s} {int(m.sum()):6d}  ({m.mean():.1%})")

b4 = P[P.model == "b4s"]
print(f"\n5) B4 CONFIDENCE (max softmax): correct {b4[~b4.err].msp.mean():.3f} | wrong {b4[b4.err].msp.mean():.3f}"
      f" | errors made with >0.9 confidence: {(b4[b4.err].msp > 0.9).mean():.1%}")

print("\n6) B4 ERROR RATE BY SCENE:", b4.groupby("group").err.mean().round(3).to_dict())
wr = b4[b4.err]
pairs = pd.Series([f"{cls[a]} -> {cls[b]}" for a, b in zip(wr.true, wr.pred)])
print("   top confusions (true -> predicted), pooled over folds and seeds:")
print(pairs.value_counts().head(12).to_string())

n1 = t[(t.block == "N1") & (t.metric == "acc")]
print(f"\n7) N1 CONTROL: {len(n1)} folds | mean {n1.value.mean():.3f} | max {n1.value.max():.3f} (chance ~0.03)")
