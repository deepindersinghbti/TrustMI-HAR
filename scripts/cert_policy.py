"""Certificate v1d: the ACT step. use / drop (hand over to the option-A backup) / abstain.

Design fixed before any policy result existed (see the chat plan of 2026-09-24):
  trust score  COMBINED v2 = logistic mix of (1 - msp, expert JS disagreement, lowest expert support),
               fitted on each fold's VALIDATION person to separate lies (s_swap) from everything else
  naming       UTD-MHAD: odd one out (argmax exp_div). MMAct: none (two sensors, naming at chance), so the
               MMAct policy is use/abstain only; a MMAct drop variant (least confident expert) is shown for
               transparency and not claimed
  use vs drop  one threshold per fold, chosen on the validation person to maximise accuracy (ties -> fewer drops)
  abstain      answer only the least suspicious 100 / 90 / 80 % of clips; compared with abstaining on
               low confidence (1 - msp), on the full mixed stream and on normal clips only
Every number is per TEST subject (seeds pooled), then mean [95% CI] across subjects.
Backups come from configs/backups.json. Writes results/certificate/v4_policy_*.csv
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cert_score import fmt, fit_logreg, load as load_merged

RES = ROOT / "results/certificate"
BACKUPS = json.load(open(ROOT / "configs/backups.json"))
COVERAGES = (1.0, 0.9, 0.8)
EXP = {"d1": ["imu", "skel", "video"], "d2": ["watch", "phone"]}


def load(ds):
    if ds == "d1":
        df = pd.read_parquet(RES / "experts_d1_all_s0-1-2_att-imu-skel-video.parquet")
        return df.assign(pred=df.pred_fused)
    return load_merged("d2", True)                  # signals (y, msp, ...) + expert columns


def backup_col(model):
    return f"exp_pred_{model[3:]}" if model.startswith("b1_") else f"model_pred_{model}"


def features(d, ds):
    sup = d[[f"exp_support_{m}" for m in EXP[ds]]].to_numpy()
    return np.column_stack([1 - d.msp.to_numpy(), d.exp_js.to_numpy(), 1 - sup.min(1)])


def namer(d, ds, variant):
    cols = [f"exp_div_{m}" for m in EXP[ds]] if variant == "odd" else [f"exp_conf_{m}" for m in EXP[ds]]
    idx = d[cols].to_numpy().argmax(1) if variant == "odd" else d[cols].to_numpy().argmin(1)
    return np.array(EXP[ds])[idx]


def backup_answer(d, ds, named):
    out = np.empty(len(d), dtype=int)
    for sen in EXP[ds]:
        k = named == sen
        out[k] = d[backup_col(BACKUPS[ds][sen])].to_numpy()[k]
    return out


def group(cond):
    return np.where(cond == "clean", "normal", np.where(cond.str.startswith("s_swap"), "lie", "benign change"))


def run_fold(va, te, ds, variant):
    Xv, Xt = features(va, ds), features(te, ds)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
    w = fit_logreg((Xv - mu) / sd, va.condition.str.startswith("s_swap").to_numpy().astype(float))
    Dv = np.c_[np.ones(len(va)), (Xv - mu) / sd] @ w
    Dt = np.c_[np.ones(len(te)), (Xt - mu) / sd] @ w
    fused_v, fused_t, yv, yt = va.pred.to_numpy(), te.pred.to_numpy(), va.y.to_numpy(), te.y.to_numpy()
    if variant is None:                             # use / abstain only
        return Dt, fused_t, np.inf
    bv = backup_answer(va, ds, namer(va, ds, variant))
    bt = backup_answer(te, ds, namer(te, ds, variant))
    cand = np.r_[np.inf, np.quantile(Dv, np.linspace(1, 0, 101))]     # from "never drop" downwards
    acc = [np.mean(np.where(Dv >= t, bv, fused_v) == yv) for t in cand]
    t = cand[int(np.argmax(acc))]                   # first maximum = the fewest drops among ties
    return Dt, np.where(Dt >= t, bt, fused_t), t


def selective(score, answer, y, cov):
    n = max(1, int(round(cov * len(y))))
    keep = np.argsort(score, kind="stable")[:n]
    return float(np.mean(answer[keep] == y[keep]))


def evaluate(df, ds, variant, label):
    df = df[df.available]
    rows = []
    for fold, g in df.groupby("fold"):
        va, te = g[g.part == "val"], g[g.part == "test"]
        Dt, final, t = run_fold(va, te, ds, variant)
        y, fused, grp = te.y.to_numpy(), te.pred.to_numpy(), group(te.condition)
        drop_flag = (Dt >= t) if variant else np.zeros(len(te), bool)
        r = dict(policy=label, dataset=ds, fold=fold, subject=te.subject.iloc[0])
        for gname in ("normal", "lie", "benign change"):
            k = grp == gname
            r[f"acc_main_{gname}"] = np.mean(fused[k] == y[k])
            r[f"acc_policy_{gname}"] = np.mean(final[k] == y[k])
            r[f"dropped_{gname}"] = np.mean(drop_flag[k])
        for stream, k in (("all", np.ones(len(te), bool)), ("normal", grp == "normal")):
            msp_score = 1 - te.msp.to_numpy()[k]
            for cov in COVERAGES:
                r[f"{stream}_cov{int(cov * 100)}_msp_main"] = selective(msp_score, fused[k], y[k], cov)
                r[f"{stream}_cov{int(cov * 100)}_cert_policy"] = selective(Dt[k], final[k], y[k], cov)
        rows.append(r)
    return pd.DataFrame(rows)


def report(res):
    for (label, ds), r in res.groupby(["policy", "dataset"], sort=False):
        print(f"\n=== {ds}: {label}  ({len(r)} test subjects)")
        print("  accuracy with no abstention (%):       main model  ->  policy        paired gain, people better")
        for gname in ("normal", "lie", "benign change"):
            d = r[f"acc_policy_{gname}"] - r[f"acc_main_{gname}"]
            print(f"    {gname:13s} {fmt(r[f'acc_main_{gname}'], 100, 1):28s} -> {fmt(r[f'acc_policy_{gname}'], 100, 1):28s}"
                  f" {fmt(d, 100, 1):28s} {int((d > 0).sum())}/{len(d)} (drop rate {100 * r[f'dropped_{gname}'].mean():.1f}%)")
        print("  abstaining (%):  answer only the least suspicious share of clips")
        for stream in ("all", "normal"):
            for cov in COVERAGES:
                a, b = r[f"{stream}_cov{int(cov * 100)}_msp_main"], r[f"{stream}_cov{int(cov * 100)}_cert_policy"]
                d = b - a
                print(f"    {stream:6s} stream, answer {int(cov * 100):3d}%: confidence {fmt(a, 100, 1):26s} "
                      f"certificate {fmt(b, 100, 1):26s} gain {fmt(d, 100, 1):26s} {int((d > 0).sum())}/{len(d)}")


def main():
    pd.set_option("display.width", 200)
    res = pd.concat([evaluate(load("d1"), "d1", "odd", "use / drop (odd one out) / abstain"),
                     evaluate(load("d2"), "d2", None, "use / abstain (official)"),
                     evaluate(load("d2"), "d2", "least_conf", "use / drop (least confident expert) / abstain [transparency only]")],
                    ignore_index=True)
    report(res)
    res.to_csv(RES / "v4_policy_per_subject.csv", index=False)
    print(f"\nsaved: {RES.relative_to(ROOT)}/v4_policy_per_subject.csv")


if __name__ == "__main__":
    main()
