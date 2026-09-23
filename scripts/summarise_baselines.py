import glob
import pandas as pd

df = pd.concat(pd.read_parquet(f) for f in glob.glob("results/raw/*_d1_*.parquet"))
t = df[(df.part == "test") & (df.condition == "clean")]

base = t[t.block == "BASE"]
per_fold = base.groupby(["model", "metric", "fold"]).value.mean().reset_index()
s = per_fold.groupby(["model", "metric"]).value.agg(["mean", "std"]).unstack("metric").round(3)
print("BASELINES (mean over folds of seed-averaged scores)")
print(s[[("mean", "acc"), ("std", "acc"), ("mean", "macro_f1"), ("std", "macro_f1")]])
print("\nruns per model:", base[base.metric == "acc"].groupby("model").size().to_dict())

print("\nB4 test accuracy per fold (seed mean):")
b4 = per_fold[(per_fold.model == "b4") & (per_fold.metric == "acc")]
print(b4.set_index("fold").value.round(3).to_string())

errs = []
for f in glob.glob("runs/BASE_d1_*/preds_test.parquet"):
    p = pd.read_parquet(f)
    errs.append(dict(model=p.exp_id.iloc[0].split("_")[2] + "_" + p.exp_id.iloc[0].split("_")[3]
                     if "b1" in p.exp_id.iloc[0] else "b4",
                     n=len(p), errors=int((p.true != p.pred).sum())))
e = pd.DataFrame(errs).groupby("model")[["n", "errors"]].sum()
e["error_rate"] = (e.errors / e.n).round(4)
print("\nPooled test errors across all folds and seeds:")
print(e)

n1 = t[(t.block == "N1") & (t.metric == "acc")]
print(f"\nN1 shuffled-label control: {len(n1)} folds, test acc "
      f"mean {n1.value.mean():.3f}, max {n1.value.max():.3f} (chance 0.037)")
