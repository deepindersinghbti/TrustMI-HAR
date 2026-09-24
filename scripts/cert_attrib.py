"""Certificate v1c: UTD-MHAD with lies planted in EVERY sensor (IMU, skeleton, video).

  (1) DETECT: can the expert signs tell a lie from benign clips, whichever sensor carries the lie?
  (2) NAME:   can they point at the lying sensor? Chance = 1/3 with three sensors.
All choices (combined-score weights, naming rule) are fitted on each fold's VALIDATION person and applied
unchanged to its TEST person; one value per test subject, then mean [95% CI] across subjects.
Reads the stand-alone file from:  cert_experts.py --dataset d1 --attack imu,skel,video
Writes results/certificate/v3_attrib_*.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cert_score import auroc, fmt, fit_logreg

RES = ROOT / "results/certificate"
FILE = RES / "experts_d1_all_s0-1-2_att-imu-skel-video.parquet"
EXP = np.array(["imu", "skel", "video"])
BASE = "1 - msp (baseline)"
COMB = [BASE, "X experts disagree (JS)", "X lowest expert support"]
RULES = {
    "odd one out (argmax exp_div)": lambda d: d[[f"exp_div_{m}" for m in EXP]].to_numpy().argmax(1),
    "least supports the answer (argmin exp_support)": lambda d: d[[f"exp_support_{m}" for m in EXP]].to_numpy().argmin(1),
    "least confident expert (argmin exp_conf)": lambda d: d[[f"exp_conf_{m}" for m in EXP]].to_numpy().argmin(1),
}


def signs(d):
    sup = d[[f"exp_support_{m}" for m in EXP]].to_numpy()
    return {BASE: 1 - d.msp.to_numpy(), "X experts disagree (JS)": d.exp_js.to_numpy(),
            "X lowest expert support": 1 - sup.min(1), "X mean expert support": 1 - sup.mean(1)}


def detection(df):
    rows = []
    for fold, g in df[df.available].groupby("fold"):
        va, te = g[g.part == "val"], g[g.part == "test"]
        yv = va.condition.str.startswith("s_swap").to_numpy()
        sv, st = signs(va), signs(te)
        Xv = np.column_stack([sv[k] for k in COMB]); mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
        w = fit_logreg((Xv - mu) / sd, yv.astype(float))
        st["COMBINED v2 (experts+msp)"] = np.c_[np.ones(len(te)), (np.column_stack([st[k] for k in COMB]) - mu) / sd] @ w
        lie = te.condition.str.startswith("s_swap").to_numpy()
        for target in ["any", *EXP]:
            pos = lie if target == "any" else (te.condition == f"s_swap:{target}").to_numpy()
            keep = pos | ~lie                            # this sensor's lies vs all benign clips
            for name, sc in st.items():
                rows.append(dict(fold=fold, subject=te.subject.iloc[0], lie_in=target, score=name,
                                 auroc=auroc(sc[keep], pos[keep])))
    return pd.DataFrame(rows)


def summarise(det):
    out = []
    for target, g in det.groupby("lie_in", sort=False):
        w = g.pivot_table(index="subject", columns="score", values="auroc")
        for name in w.columns:
            d = (w[name] - w[BASE]).dropna()
            out.append(dict(lie_in=target, score=name, auroc=fmt(w[name]),
                            minus_msp=fmt(d) if name != BASE else "",
                            subjects_better=f"{int((d > 0).sum())}/{len(d)}" if name != BASE else ""))
    return pd.DataFrame(out)


def naming(df):
    rows, conf = [], []
    lies = df[df.available & df.condition.str.startswith("s_swap")]
    for fold, g in lies.groupby("fold"):
        va, te = g[g.part == "val"], g[g.part == "test"]
        acc_val = {r: (EXP[f(va)] == va.attacked.to_numpy()).mean() for r, f in RULES.items()}
        best = max(acc_val, key=acc_val.get)
        for r, f in RULES.items():
            named = EXP[f(te)]
            if r == best:
                conf.append(pd.crosstab(te.attacked, named).reindex(index=EXP, columns=EXP, fill_value=0))
            for sen in EXP:
                k = te.attacked.to_numpy() == sen
                rows.append(dict(fold=fold, subject=te.subject.iloc[0], rule=r, chosen=(r == best),
                                 lie_in=sen, acc=(named[k] == sen).mean()))
    return pd.DataFrame(rows), sum(conf)


def main():
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
    df = pd.read_parquet(FILE)
    print(f"{FILE.name}: {len(df)} rows, conditions: {sorted(df.condition.unique())}")
    summ = summarise(detection(df))
    print("\nDETECTION: AUROC per test subject, mean [95% CI] (this sensor's lies vs all benign clips)")
    print(summ.to_string(index=False))
    nm, conf = naming(df)
    ch = nm[nm.chosen]
    print("\nNAMING the lying sensor (chance 33.3%), rule chosen on validation:")
    print(f"  overall   {fmt(ch.groupby('subject').acc.mean(), 100, 1)}")
    for sen, g in ch.groupby("lie_in", sort=False):
        print(f"  lie in {sen:5s} {fmt(g.acc, 100, 1)}")
    print("rules chosen per fold:", ch.drop_duplicates("fold").rule.value_counts().to_dict())
    print("\nwho gets blamed, rule chosen on validation (rows: lying sensor, % of its lies; all folds and seeds):")
    print((100 * conf.div(conf.sum(axis=1), axis=0)).round(1).to_string())
    print("\neach rule on test, overall, for transparency only (not used for any choice):")
    print(nm.groupby(["rule", "subject"]).acc.mean().groupby("rule").apply(lambda v: fmt(v, 100, 1)).to_string())
    summ.to_csv(RES / "v3_attrib_detection_summary.csv", index=False)
    nm.to_csv(RES / "v3_attrib_naming_per_subject.csv", index=False)
    print(f"\nsaved: {RES.relative_to(ROOT)}/v3_attrib_*.csv")


if __name__ == "__main__":
    main()
