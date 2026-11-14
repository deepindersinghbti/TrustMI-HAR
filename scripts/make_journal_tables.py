"""Part C.2 of docs/plan_followup_2026-11-12.md: every journal table and figure, built by script from the frozen
per-person / per-fold result files. No number is typed by hand.

  python scripts/make_journal_tables.py                 -> paper/journal/{tables,figs}/ + paper/journal/NUMBERS.md
  python scripts/make_journal_tables.py --out /tmp/x    -> same, somewhere else (reproducibility check)

Inputs (all frozen, see results/FROZEN_MANIFEST.sha256): results/certificate/*_per_subject.csv, *_per_fold.csv,
*_per_unit.csv, imuoff_d3_predictions.csv, v1_*_auroc_summary.csv, mi_bound_summary.csv,
realdrop_d3_rows.parquet, nowatch_d2v_per_subject.csv; runs/BASE_*/preds_test.parquet (model accuracy only).

Statistics, exactly as the scorers: unit = test person (UTD-MHAD 8, MMAct 20) or fold (WearGait-PD 10);
mean over units; 95% interval = mean +/- t(0.975, n-1) * sd / sqrt(n); "better" = units with a paired difference > 0.
Revision 1 is the main result everywhere; revision 2 and the post hoc rules are only shown where labelled.

CHECKS (the script stops if any fails): every headline number is re-read from the scorer's own log and must agree
to the precision printed there (logs/cert_mi_score_d1v2.log, cert_mi_score_d2v.log, score_mi_d3.log, the three fault
logs, score_dropout_d3.log, score_realdrop_d3.log, cert_nowatch_score_d2v.log), and the IMU-off differences must
equal results/certificate/imuoff_d3_predictions.csv.
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cert_score import tq        # the scorers' own t quantile, so the intervals are computed identically
RES, LOGS, RUNS = ROOT / "results/certificate", ROOT / "logs", ROOT / "runs"
DS = ("d1v2", "d2v", "d3")
DSNAME = {"d1v2": "UTD-MHAD", "d2v": "MMAct", "d3": "WearGait-PD"}
UNIT = {"d1v2": "person", "d2v": "person", "d3": "fold"}
UNITS = {"d1v2": "people", "d2v": "people", "d3": "folds"}
SENS = {"d1v2": ("imu", "skel", "video"), "d2v": ("watch", "phone", "video"), "d3": ("arms", "trunk", "legs")}
SNAME = {"imu": "IMU", "skel": "skeleton", "video": "video", "watch": "watch", "phone": "phone",
         "arms": "arms", "trunk": "trunk", "legs": "legs"}
MAIN = "REV3 = C3 + MI (main)"
EV = {"d1v2": ("TMC", "ECML"), "d2v": ("TMC_S3", "ECML_S3"), "d3": ("TMC_W", "ECML_W")}
LIE_ROWS = ("C3 (paper certificate)", "MI only", "s1 + MI (no witnesses)", MAIN, "REV3 frozen MI (ablation)")
LABEL = {"C3 (paper certificate)": "C3 (conference certificate)", "MI only": "MI only",
         "s1 + MI (no witnesses)": "Stability + MI (no witnesses)", MAIN: "REV3 (ours, main)",
         "REV3 frozen MI (ablation)": "REV3, frozen MI encoders", "confidence alone": "Confidence (MSP)",
         "C3": "C3 (conference certificate)", "REV3 (main)": "REV3 (ours, main)", "D (dropout model)": "Dropout-trained model",
         "REV3 known faults": "REV3, faults in the fit [extra]"}
for d in DS:
    for e, nm in zip(EV[d], ("TMC", "ECML")):
        LABEL[f"{e} alone"] = nm
LOGFILE = {"lies": {"d1v2": "cert_mi_score_d1v2.log", "d2v": "cert_mi_score_d2v.log", "d3": "score_mi_d3.log"},
           "faults": {"d1v2": "cert_fault_score_d1v2.log", "d2v": "cert_fault_score_d2v.log", "d3": "score_fault_d3.log"}}
BASE_MODELS = {  # (role, run-name model); backups as in configs/backups.json
    "d1v2": [("Main fused model (IMU + skeleton + video)", "b4"), ("IMU alone", "b1_imu"), ("Skeleton alone", "b1_skel"),
             ("Video alone", "b1_video"), ("Without IMU", "b4_noimu"), ("Without skeleton", "b4_noskel"),
             ("Without video", "b4_novideo"), ("Dropout-trained (p=0.5)", "b4_sd50"), ("TMC", "tmc"), ("ECML", "ecml")],
    "d2v": [("Main fused model (watch + phone + video)", "b4s3"), ("Watch alone", "b1_watch"), ("Phone alone", "b1_phone"),
            ("Video alone", "b1_video"), ("Without watch", "b4s_nowatch"), ("Without phone", "b4s_nophone"),
            ("Without video", "b4s"), ("Dropout-trained (p=0.5)", "b4s3_sd50"), ("TMC", "tmc_s3"), ("ECML", "ecml_s3")],
    "d3": [("Main fused model (arms + trunk + legs)", "b4w"), ("Arms alone", "b1_arms"), ("Trunk alone", "b1_trunk"),
           ("Legs alone", "b1_legs"), ("Without arms", "b4w_noarms"), ("Without trunk", "b4w_notrunk"),
           ("Without legs", "b4w_nolegs"), ("Dropout-trained (p=0.5)", "b4w_sd50"), ("TMC", "tmc_w"), ("ECML", "ecml_w")]}

CHECKS = []          # (what, ours, log, tolerance, ok)


# ----------------------------------------------------------------- statistics (same as the scorers)
def stats(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v)
    m = v.mean() if n else np.nan
    h = tq(n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - h, m + h, int((v > 0).sum()), n


def fmt(m, lo=None, hi=None, k=1.0, d=3, sign=False):
    if m is None or np.isnan(m):
        return "--"
    f = (lambda x: f"{k * x:+.{d}f}") if sign else (lambda x: f"{k * x:.{d}f}")
    return f(m) if lo is None or np.isnan(lo) else f"{f(m)} [{f(lo)}, {f(hi)}]"


def check(what, ours, logval, dec):
    tol = 0.5 * 10 ** (-dec) + 1e-9
    ok = (np.isnan(ours) and np.isnan(logval)) or abs(ours - logval) <= tol
    CHECKS.append((what, ours, logval, tol, ok))


# ----------------------------------------------------------------- reading the scorers' printed tables
def log_block(fname, header):
    """Lines after the first line that contains `header`, up to the next blank line."""
    lines = (LOGS / fname).read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if header in l)
    out = []
    for l in lines[i + 1:]:
        if not l.strip():
            break
        out.append(l)
    return out


def log_rows(lines, numcols):
    """'  name<2+ spaces>num num ...' -> {name: [floats]} (first `numcols` numbers after the name)."""
    rows = {}
    for l in lines:
        m = re.match(r"^\s*(.+?)\s{2,}(-?[\d.]+|nan|NaN)(\s.*)?$", l)
        if not m:
            continue
        nums = re.findall(r"-?\d+\.\d+|-?\d+|NaN|nan", l[m.start(2):])
        rows.setdefault(m.group(1).strip(), [float(x) for x in nums[:numcols]])   # first table wins
    return rows


# ----------------------------------------------------------------- output helpers
def md_table(df):
    """Markdown table without extra packages; cells exactly as written."""
    cells = [[str(c) for c in df.columns]] + [[str(v) for v in r] for r in df.itertuples(index=False)]
    w = [max(len(r[j]) for r in cells) for j in range(len(cells[0]))]
    line = lambda r: "| " + " | ".join(c.ljust(k) for c, k in zip(r, w)) + " |"
    return "\n".join([line(cells[0]), "|" + "|".join("-" * (k + 2) for k in w) + "|"] + [line(r) for r in cells[1:]])


class Out:
    def __init__(self, base):
        self.base = Path(base); (self.base / "tables").mkdir(parents=True, exist_ok=True)
        (self.base / "figs").mkdir(parents=True, exist_ok=True)
        self.md = ["# Journal numbers (built by scripts/make_journal_tables.py; do not edit by hand)", "",
                   "Mean over test units, 95% interval over units in brackets; 'better' = units where the paired "
                   "difference is above 0. Revision 1 unless labelled.", ""]

    def table(self, name, title, df, note="", col_format=None):
        df.to_csv(self.base / "tables" / f"{name}.csv", index=False)
        cols = list(df.columns)
        spec = col_format or ("l" * 2 + "c" * (len(cols) - 2) if cols and cols[0] == "Dataset" else "l" + "c" * (len(cols) - 1))
        esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("+/-", r"$\pm$")
        tex = [f"% {title}", "% built by scripts/make_journal_tables.py from frozen results; do not edit by hand",
               r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
        prev = None
        for _, r in df.iterrows():
            cells = [esc(v) for v in r.values]
            if cols[0] == "Dataset":
                if prev is not None and r.iloc[0] != prev:
                    tex.append(r"\midrule")
                if r.iloc[0] == prev:
                    cells[0] = ""
                prev = r.iloc[0]
            tex.append(" & ".join(cells) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}"]
        (self.base / "tables" / f"{name}.tex").write_text("\n".join(tex) + "\n")
        self.md += [f"## {name}: {title}", ""]
        if note:
            self.md += [note, ""]
        self.md += [md_table(df), ""]


# ----------------------------------------------------------------- T1: models
def t_models(o):
    logged = {}
    for f in sorted(LOGS.glob("*.log")):
        for m in re.finditer(r"(BASE_\S+): best epoch\s+\d+ \| test acc ([\d.]+) macro-F1 ([\d.]+)", f.read_text()):
            logged[m.group(1)] = (float(m.group(2)), float(m.group(3)))      # last occurrence wins (re-runs)
    rows = []
    for ds in DS:
        for role, mdl in BASE_MODELS[ds]:
            per_fold = {}
            for p in sorted(RUNS.glob(f"BASE_{ds}_{mdl}_loso_f*_s*/preds_test.parquet")):
                fold = int(re.search(r"_f(\d+)_s\d+", p.parent.name).group(1))
                d = pd.read_parquet(p, columns=["true", "pred"])
                y, yh = d["true"].to_numpy(), d["pred"].to_numpy()
                f1 = np.mean([2 * ((y == c) & (yh == c)).sum() / max((y == c).sum() + (yh == c).sum(), 1)
                              for c in np.unique(y)])        # as train.py: classes present in the test set
                if p.parent.name in logged:
                    check(f"T1 {p.parent.name} test acc", (y == yh).mean(), logged[p.parent.name][0], 3)
                    check(f"T1 {p.parent.name} macro-F1", f1, logged[p.parent.name][1], 3)
                per_fold.setdefault(fold, []).append(((y == yh).mean(), f1))
            if not per_fold:
                rows.append({"Dataset": DSNAME[ds], "Model": role, "Runs": 0, "Accuracy %": "missing",
                             "Macro-F1": "missing"}); continue
            acc = [np.mean([a for a, _ in v]) for v in per_fold.values()]
            f1s = [np.mean([b for _, b in v]) for v in per_fold.values()]
            a, f = stats(acc), stats(f1s)
            rows.append({"Dataset": DSNAME[ds], "Model": role, "Runs": sum(len(v) for v in per_fold.values()),
                         "Accuracy %": fmt(*a[:3], k=100, d=1), "Macro-F1": fmt(*f[:3], d=3)})
    o.table("T1_models", "Clean test accuracy of every trained model (folds x 3 seeds; seeds averaged within a "
            "fold, interval over folds)", pd.DataFrame(rows),
            "Macro-F1 as in scripts/train.py: the mean F1 over the activities present in the test set of that fold.")


# ----------------------------------------------------------------- T2: lies
def lie_tables(o):
    rows, diffs = [], []
    for ds in DS:
        d = pd.read_csv(RES / f"mi_{ds}_per_subject.csv")
        d = d[d.revision == 1]
        systems = list(LIE_ROWS) + [f"{e} alone" for e in EV[ds]]
        logr = log_rows(log_block(LOGFILE["lies"][ds], f"{ds}: certificate on B4, revision 1"), 6)
        for s in systems:
            g = d[d.system == s].sort_values("fold")
            vals = {c: stats(g[c]) for c in ("lie_auroc", "naming", "acc_policy_lie", "acc_policy_normal",
                                              "error_auroc", "all_cov80_cert_policy")}
            lv = logr[s]
            for j, (c, k, dec) in enumerate((("lie_auroc", 1, 3), ("error_auroc", 1, 3), ("naming", 100, 1),
                                             ("acc_policy_lie", 100, 1), ("acc_policy_normal", 100, 1),
                                             ("all_cov80_cert_policy", 100, 1))):
                check(f"lies {ds} {s} {c}", k * vals[c][0], lv[j], dec)
            rows.append({"Dataset": DSNAME[ds], "System": LABEL[s],
                         "Lie AUROC": fmt(*vals["lie_auroc"][:3]),
                         "Names the lying sensor %": fmt(*vals["naming"][:3], k=100, d=1),
                         "Accuracy on lie clips % (after use/drop)": fmt(vals["acc_policy_lie"][0], k=100, d=1),
                         "Accuracy on normal clips %": fmt(vals["acc_policy_normal"][0], k=100, d=1),
                         "Error AUROC (natural mistakes)": fmt(vals["error_auroc"][0])})
        m = d[d.system == MAIN].sort_values("fold").set_index("fold")
        for s in ("C3 (paper certificate)", f"{EV[ds][0]} alone", f"{EV[ds][1]} alone"):
            g = d[d.system == s].sort_values("fold").set_index("fold")
            r = {"Dataset": DSNAME[ds], "REV3 minus": LABEL[s]}
            for c, nm, k, dec in (("lie_auroc", "Lie AUROC", 1, 3), ("naming", "Naming (points)", 100, 1),
                                  ("acc_policy_lie", "Lie-clip accuracy (points)", 100, 1),
                                  ("acc_policy_normal", "Normal-clip accuracy (points)", 100, 1)):
                st = stats(m[c] - g[c])
                r[nm] = fmt(*st[:3], k=k, d=dec, sign=True) + f" ({st[3]}/{st[4]})"
            diffs.append(r)
    o.table("T2_lies", "Planted lies: detection, naming and accuracy after use / drop (certificate on the main "
            "model, revision 1)", pd.DataFrame(rows),
            "Lie AUROC and naming with 95% intervals over units (UTD-MHAD 8 people, MMAct 20 people, WearGait-PD "
            "10 folds).")
    o.table("T2b_lies_paired", "Planted lies: REV3 minus each other system, paired per unit (units better / all)",
            pd.DataFrame(diffs))

    # certificate placed on TMC / ECML
    rows = []
    for ds in DS:
        for e in EV[ds]:
            f = RES / f"mi_{ds}_on_{e.lower()}_per_subject.csv"
            d = pd.read_csv(f); d = d[d.revision == 1]
            for s in (f"{e} alone", "C3 (paper certificate)", MAIN):
                g = d[d.system == s]
                rows.append({"Dataset": DSNAME[ds], "Answering model": LABEL[f"{e} alone"], "System": LABEL[s],
                             "Lie AUROC": fmt(*stats(g.lie_auroc)[:3]),
                             "Names the lying sensor %": fmt(stats(g.naming)[0], k=100, d=1),
                             "Accuracy on lie clips %": fmt(stats(g.acc_policy_lie)[0], k=100, d=1),
                             "Accuracy on normal clips %": fmt(stats(g.acc_policy_normal)[0], k=100, d=1)})
    o.table("T3_lies_on_evidential", "The certificate placed on TMC and on ECML (planted lies, revision 1)",
            pd.DataFrame(rows))

    # naming confusion, REV3: one table per dataset (each has its own sensors)
    for ds in DS:
        d = pd.read_csv(RES / f"mi_{ds}_naming_by_sensor.csv")
        d = d[d.system == MAIN]
        rows = []
        for s in SENS[ds]:
            g = d[d.lie_in == s]
            r = {"Lie planted in": SNAME[s]}
            for n in SENS[ds]:
                r[f"named {SNAME[n]} %"] = fmt(g[f"named_{n}"].mean(), k=100, d=1)
            fa = pd.read_csv(RES / f"mi_{ds}_naming_by_sensor.csv")
            fa = fa[(fa.system == "C3 (paper certificate)") & (fa.lie_in == s)].fused_acc
            r["Main model alone right %"] = fmt(fa.mean(), k=100, d=1)
            rows.append(r)
        o.table(f"T4_naming_{ds}", f"{DSNAME[ds]}: which sensor REV3 names, by where the lie was planted "
                f"(% of that sensor's lie clips, mean over {UNITS[ds]})", pd.DataFrame(rows))


# ----------------------------------------------------------------- T5: faults
FAULT_NOTE = ("Faults = drift, gap (partial dropout) and timing shift, 3 levels each, in every sensor; the certificate "
              "is fitted on validation lies only and transferred unchanged ('frozen'). 'REV3, faults in the fit' is a "
              "labelled extra row and is not shown here.")


def fault_tables(o, prefix="faults", logs=None, tag="faults", names=("T5_faults", "T5b_faults_paired"),
                 titles=("Everyday sensor faults (frozen transfer, all faults pooled, revision 1)",
                         "Everyday faults: REV3 minus other systems, harmful-fault AUROC, paired per unit"),
                 note=FAULT_NOTE):
    """The fault tables of one fault bench (default: the original bench, T5 / T5b; faults2 = addition 1)."""
    logs = logs or LOGFILE["faults"]
    rows, diffs = [], []
    for ds in DS:
        d = pd.read_csv(RES / f"{prefix}_{ds}_per_subject.csv")
        d = d[(d.revision == 1) & (d.group == "all faults")]
        systems = ["confidence alone", "C3", "REV3 (main)"] + [f"{e} alone" for e in EV[ds]] + ["D (dropout model)"]
        det = log_rows(log_block(logs[ds], "DETECTION, AUROC fault clips vs clean clips"), 7)
        nam = log_rows(log_block(logs[ds], "NAMING the faulty sensor"), 6)
        hdr_det = [l for l in (LOGS / logs[ds]).read_text().splitlines()
                   if l.startswith("system ") and "confidence alone" in l][0]
        det_cols = re.split(r"\s{2,}", hdr_det.strip())[1:]
        hdr_nam = [l for l in (LOGS / logs[ds]).read_text().splitlines()
                   if l.startswith("system ") and "confidence alone" not in l][0]
        nam_cols = re.split(r"\s{2,}", hdr_nam.strip())[1:]
        for s in systems:
            g = d[d.system == s]
            st = {c: stats(g[c]) for c in ("auroc_harmful", "auroc", "naming_harmful", "acc_fault", "acc_policy_fault",
                                           "false_alarm", "flagged")}
            check(f"{tag} {ds} {s} auroc_harmful", st["auroc_harmful"][0], det["all faults"][det_cols.index(s)], 3)
            if s in nam_cols:
                check(f"{tag} {ds} {s} naming_harmful", 100 * st["naming_harmful"][0],
                      nam["all faults"][nam_cols.index(s)], 3)
            if s != "confidence alone":
                line = next(l for l in log_block(logs[ds], "RECOVERY, revision 1")
                            if re.search(r"\s" + re.escape(s) + r"\s{2,}", " " + l))
                nums = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
                check(f"{tag} {ds} {s} acc_policy_fault", 100 * st["acc_policy_fault"][0], nums[1], 3)
            rows.append({"Dataset": DSNAME[ds], "System": LABEL[s],
                         "AUROC, harmful faults": fmt(*st["auroc_harmful"][:3]),
                         "AUROC, all faults": fmt(st["auroc"][0]),
                         "Names faulty sensor % (harmful)": fmt(st["naming_harmful"][0], k=100, d=1),
                         "Fault-clip acc. %: alone -> use/drop": f"{fmt(st['acc_fault'][0], k=100, d=1)} -> "
                                                                 f"{fmt(st['acc_policy_fault'][0], k=100, d=1)}",
                         "False alarms on clean %": fmt(st["false_alarm"][0], k=100, d=1)})
        rv = d[d.system == "REV3 (main)"].set_index("fold")
        for s in ("confidence alone", "C3", f"{EV[ds][0]} alone"):
            g = d[d.system == s].set_index("fold")
            st = stats(rv.auroc_harmful - g.auroc_harmful)
            diffs.append({"Dataset": DSNAME[ds], "REV3 minus": LABEL[s],
                          "AUROC, harmful faults": fmt(*st[:3], sign=True) + f" ({st[3]}/{st[4]})"})
    o.table(names[0], titles[0], pd.DataFrame(rows), note)
    o.table(names[1], titles[1], pd.DataFrame(diffs))


# ----------------------------------------------------------------- T6: missing sensors
def rate_all(p, system, a, b):
    g = p[p.system == system]
    return g[a].sum() / g[b].sum()


def missing_tables(o):
    # MMAct, real no-watch clips
    p = pd.read_csv(RES / "nowatch_d2v_per_subject.csv"); p = p[p.revision == 1]
    big = [2, 13, 16, 20]
    logr = log_rows(log_block("cert_nowatch_score_d2v.log", "SYSTEMS, revision 1"), 10)
    rows = []
    for s, g in p.groupby("system", sort=False):
        rate = lambda a, b: g[a].sum() / g[b].sum()
        gb = g.set_index("subject").loc[big]
        v = {"acc_alone": 100 * rate("correct_alone_nw", "n_nw"), "acc_policy": 100 * rate("correct_policy_nw", "n_nw"),
             "ref": 100 * rate("correct_policy_ref", "n_ref"), "flagged": 100 * rate("n_flagged_nw", "n_nw"),
             "fa": 100 * rate("n_flagged_ref", "n_ref"),
             "named": np.nan if s == "confidence alone" else 100 * rate("n_named_watch_flagged_nw", "n_flagged_nw"),
             "auroc": gb.auroc_nw.mean()}
        lv = logr[s]
        for j, c in ((0, "acc_alone"), (1, "acc_policy"), (3, "flagged"), (5, "named")):
            check(f"nowatch {s} {c}", v[c], lv[j], 3)
        check(f"nowatch {s} auroc", v["auroc"], lv[7], 3)
        rows.append({"System": LABEL.get(s, s), "Acc. alone %": fmt(v["acc_alone"], d=1),
                     "Acc. after use/drop %": fmt(v["acc_policy"], d=1), "Flagged %": fmt(v["flagged"], d=1),
                     "Names the watch (flagged) %": fmt(v["named"], d=1),
                     "AUROC no-watch vs reference (4 people)": fmt(v["auroc"]),
                     "Acc. on reference clips %": fmt(v["ref"], d=1)})
    km = re.search(r"pooled\s*: doing nothing\s+([\d.]+) \| REV3 policy\s+([\d.]+) \| known missing\s+([\d.]+)",
                   (LOGS / "cert_nowatch_score_d2v.log").read_text())
    check("nowatch pooled doing nothing (log line)", 100 * rate_all(p, "confidence alone", "correct_alone_nw", "n_nw"),
          float(km.group(1)), 1)
    rows.append({"System": "Backup without the watch, sensor known missing [reference]", "Acc. alone %": "--",
                 "Acc. after use/drop %": km.group(3), "Flagged %": "--", "Names the watch (flagged) %": "--",
                 "AUROC no-watch vs reference (4 people)": "--", "Acc. on reference clips %": "--"})
    o.table("T6_nowatch", "MMAct: 247 real clips without a watch signal (6 people, 3 seeds pooled; revision 1)",
            pd.DataFrame(rows), "The watch channels are filled with the training mean. 'Acc. alone' of the first row "
            "= doing nothing. The reference row (an oracle that knows the watch is missing) is read from the "
            "scorer's log, cert_nowatch_score_d2v.log, where it is computed from the stored model outputs.")

    # WearGait-PD real dropouts
    pf = pd.read_csv(RES / "realdrop_d3_per_fold.csv"); pf = pf[pf.revision == 1]
    R = pd.read_parquet(RES / "realdrop_d3_rows.parquet")
    R = R[R.revision == 1]
    logr = log_rows(log_block("score_realdrop_d3.log", "SYSTEMS, revision 1"), 8)
    rows = []
    for s, g in R.groupby("system", sort=False):
        r1, ref = g[g.set == "rd1"], g[g.set == "ref"]
        fl = r1[r1.flagged]
        q = pf[pf.system == s]
        v = {"alone": 100 * (r1.pred == r1.y).mean(), "policy": 100 * (r1.final == r1.y).mean(),
             "flag": 100 * r1.flagged.mean(), "fa": 100 * ref.flagged.mean(),
             "name": 100 * (fl.named == fl.group).mean() if len(fl) and s != "confidence alone" else np.nan,
             "auroc": q.auroc.mean()}
        lv = logr[s]
        for j, c in ((0, "alone"), (1, "policy"), (3, "flag"), (4, "fa"), (5, "name"), (6, "auroc")):
            check(f"realdrop {s} {c}", v[c], lv[j], 3)
        rows.append({"System": LABEL.get(s, s), "Acc. alone %": fmt(v["alone"], d=1),
                     "Acc. after use/drop %": fmt(v["policy"], d=1), "Flagged %": fmt(v["flag"], d=1),
                     "False alarms (clean) %": fmt(v["fa"], d=1), "Names right group (flagged) %": fmt(v["name"], d=1),
                     "AUROC dropout vs clean (9 folds)": fmt(v["auroc"])})
    km = re.search(r"pooled: doing nothing\s+([\d.]+) \| REV3 policy\s+([\d.]+) \| known missing\s+([\d.]+)",
                   (LOGS / "score_realdrop_d3.log").read_text())
    rv = R[(R.system == "REV3 (main)") & (R.set == "rd1")]
    check("realdrop pooled REV3 policy (log line)", 100 * (rv.final == rv.y).mean(), float(km.group(2)), 1)
    rows.append({"System": "Backup for the missing group, known missing [reference]", "Acc. alone %": "--",
                 "Acc. after use/drop %": km.group(3), "Flagged %": "--", "False alarms (clean) %": "--",
                 "Names right group (flagged) %": "--", "AUROC dropout vs clean (9 folds)": "--"})
    m = pf[pf.system == "REV3 (main)"].sort_values("fold")
    st = stats(100 * (m.acc_policy_rd - m.acc_alone_rd))
    w = (m.acc_policy_rd - m.acc_alone_rd).values
    main = (f"Main result named in advance (REV3 use / drop minus doing nothing, single-group dropout clips): "
            f"{st[0]:+.1f} points [{st[1]:+.1f}, {st[2]:+.1f}] over 10 folds; better in {int((w > 0).sum())}, "
            f"level in {int((w == 0).sum())}, worse in {int((w < 0).sum())}.")
    o.table("T7_realdrop", "WearGait-PD: real sensor dropouts (3,250 single-group clips x 3 seeds, pooled; revision 1)",
            pd.DataFrame(rows), main)

    # IMU-off test
    d = pd.read_csv(RES / "imuoff_d3_per_fold.csv"); d = d[(d.revision == 1) & (d.group != "clean")]
    pred = pd.read_csv(RES / "imuoff_d3_predictions.csv")
    rows, curve = [], []
    for (grp, k), g in d.groupby(["group", "k"], sort=False):
        rv = g[g.system == "REV3 (main)"].sort_values("fold")
        st = stats(100 * (rv.acc_policy.values - rv.acc_alone.values))
        pr = pred[(pred["item"] == "main") & (pred.group == grp) & (pred.k == k)]
        if len(pr):
            check(f"imuoff {grp} k={k} REV3 minus nothing (points)", st[0], pr["mean"].iloc[0], 9)
        r = {"Group": grp, "IMUs off": int(k), "Doing nothing %": fmt(rv.acc_alone.mean(), k=100, d=1)}
        for s in ("REV3 (main)", "TMC_W alone", "D (dropout model)"):
            r[f"{LABEL[s]} %"] = fmt(g[g.system == s].acc_policy.mean(), k=100, d=1)
        r["REV3 minus nothing (points)"] = fmt(*st[:3], d=2, sign=True)
        r["Folds better"] = f"{st[3]}/{st[4]}"
        r["REV3 AUROC"] = fmt(rv.auroc.mean())
        rows.append(r)
        curve.append((grp, int(k), *st[:3]))
    o.table("T8_imuoff", "WearGait-PD IMU-off test: k IMUs of one body group switched off for the whole clip "
            "(planned test; mean over 10 folds; revision 1)", pd.DataFrame(rows),
            "Predictions P1-P5 were written before the run (docs/plan_followup_2026-11-12.md, A.6); all five were "
            "confirmed. k = all IMUs equals the simulated removal r_fill.")
    return pd.DataFrame(curve, columns=["group", "k", "mean", "lo", "hi"])


# ----------------------------------------------------------------- T9: dropout model, T10: post hoc, T11: v1, T12: MI
def other_tables(o):
    rows = []
    for ds in DS:
        d = pd.read_csv(RES / f"dropout_{ds}_per_subject.csv"); d = d[d.revision == 1]
        a = d[d.system == "A main (separate backups)"].set_index("fold")
        for s, g in d.groupby("system", sort=False):
            if "sd25" in s:
                continue
            g = g.set_index("fold")
            st = stats(100 * (g.acc_policy_lie - a.acc_policy_lie)) if s[0] != "A" else None
            rows.append({"Dataset": DSNAME[ds], "Backup design": s.split(" ", 1)[1],
                         "Lie AUROC": fmt(g.lie_auroc.mean()),
                         "Normal acc. alone %": fmt(g.acc_main_normal.mean(), k=100, d=1),
                         "Lie-clip acc. after use/drop %": fmt(g.acc_policy_lie.mean(), k=100, d=1),
                         "Minus separate backups (points)": "--" if st is None else
                         fmt(*st[:3], d=1, sign=True) + f" ({st[3]}/{st[4]})"})
    dd = pd.DataFrame(rows)
    lg = log_rows(log_block("score_dropout_d3.log", "sensor-dropout ablation, revision 1"), 7)
    d3 = pd.read_csv(RES / "dropout_d3_per_subject.csv"); d3 = d3[d3.revision == 1]
    for s, g in d3.groupby("system"):
        check(f"dropout d3 {s} policy lie", 100 * g.acc_policy_lie.mean(), lg[s][5], 1)
    o.table("T9_dropout", "Separate backup models vs one dropout-trained model (planted lies, revision 1)", dd,
            "B = hide the named sensor in the main model without retraining; C = dropout-trained model (p=0.5) as "
            "the backup; D = dropout-trained model used everywhere.")

    p = pd.read_csv(RES / "posthoc_per_unit.csv")
    rows = []
    spec = {"lies": [("lie_auroc", "Lie AUROC", 1, 3), ("acc_policy_lie", "Lie-clip acc. %", 100, 1)],
            "faults": [("auroc_harmful", "Harmful-fault AUROC", 1, 3), ("acc_policy_fault", "Fault-clip acc. %", 100, 1)],
            "nowatch": [("auroc_nw_mean4", "No-watch AUROC", 1, 3), ("acc_policy_nw", "No-watch acc. %", 100, 1)],
            "realdrop": [("auroc_rd1", "Dropout AUROC", 1, 3), ("acc_policy_rd1", "Dropout acc. %", 100, 1)]}
    for (bench, ds), g in p.groupby(["bench", "dataset"], sort=False):
        if bench not in spec:
            continue
        for v, h in g.groupby("variant", sort=False):
            r = {"Bench": bench, "Dataset": DSNAME[ds], "Variant": v}
            for j, (c, nm, k, dec) in enumerate(spec[bench]):
                r[f"measure {j + 1}"] = f"{nm}: {fmt(h[c].mean(), k=k, d=dec)}"
            rows.append(r)
    o.table("T10_posthoc", "POST HOC rule changes (designed after seeing results; labelled extra rows only)",
            pd.DataFrame(rows), "Rule (a): confidence as a second alarm (detection and abstention only). Rule (b): "
            "drop only when the naming margin passes a threshold chosen on validation lies. Means over units. IMU-off "
            "rows: results/certificate/posthoc_imuoff_per_fold.csv.")

    rows = []
    for ds in DS:
        v = pd.read_csv(RES / f"v1_{ds}_auroc_summary.csv")
        v = v.drop(columns="dataset"); v.insert(0, "Dataset", DSNAME[ds]); rows.append(v.fillna("--"))
    o.table("T11_v1_signs", "Ablation: the conference warning signs (v1) next to the journal signals (copied from the "
            "scorer's summary files)", pd.concat(rows))

    mb = pd.read_csv(RES / "mi_bound_summary.csv")
    mb = mb[mb.dataset.isin(DS) & (mb.version == "own")]
    mb = pd.DataFrame({"Dataset": mb.dataset.map(DSNAME), "Sensor pair": mb.pair,
                       "MI lower bound (nats)": [fmt(a, b, c, d=2) for a, b, c in zip(mb["mean"], mb.lo, mb.hi)],
                       "Folds": mb.n_folds, "Range": [f"{a:.2f} to {b:.2f}" for a, b in zip(mb.min_fold, mb.max_fold)]})
    o.table("T12_mi_bound", "Dataset-level mutual information between sensors (InfoNCE lower bound, own matcher)", mb)


# ----------------------------------------------------------------- figures
def figures(o, curve):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    # F1: per-unit lie AUROC
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=False)
    for ax, ds in zip(axs, DS):
        d = pd.read_csv(RES / f"mi_{ds}_per_subject.csv"); d = d[d.revision == 1]
        sy = ["C3 (paper certificate)", f"{EV[ds][0]} alone", f"{EV[ds][1]} alone", MAIN]
        piv = d.pivot(index="fold", columns="system", values="lie_auroc")[sy]
        x = np.arange(len(sy))
        for _, r in piv.iterrows():
            ax.plot(x, r.values, color="0.75", lw=0.6, zorder=1)
        ax.scatter(np.repeat(x[None], len(piv), 0).ravel(), piv.values.ravel(), s=7, color="0.3", zorder=2)
        ax.plot(x, piv.mean().values, "o-", color="C0", ms=4, lw=1.2, zorder=3)
        ax.set_xticks(x, ["C3", "TMC", "ECML", "REV3"]); ax.set_title(f"{DSNAME[ds]} ({len(piv)} {UNITS[ds]})")
        ax.set_ylabel("Lie AUROC" if ds == "d1v2" else "")
    fig.tight_layout(); fig.savefig(o.base / "figs/F1_lie_auroc_per_unit.pdf"); fig.savefig(o.base / "figs/F1_lie_auroc_per_unit.png", dpi=200)
    plt.close(fig)
    # F2: IMU-off curve
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for grp, mk, dx in (("legs", "o", 0.0), ("trunk", "s", 0.12), ("arms", "^", -0.12)):
        c = curve[curve.group == grp].sort_values("k")
        ax.errorbar(c.k + dx, c["mean"], yerr=[c["mean"] - c.lo, c.hi - c["mean"]], fmt=mk + "-", ms=3.5, lw=1, capsize=2,
                    label=grp)
    ax.axhline(0, color="0.5", lw=0.6, ls="--")
    ax.set_xlabel("IMUs of the group switched off (k)"); ax.set_ylabel("REV3 minus doing nothing\n(accuracy points)")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(o.base / "figs/F2_imuoff.pdf"); fig.savefig(o.base / "figs/F2_imuoff.png", dpi=200); plt.close(fig)
    # F3: fault detection (harmful) per system
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=True)
    for ax, ds in zip(axs, DS):
        d = pd.read_csv(RES / f"faults_{ds}_per_subject.csv")
        d = d[(d.revision == 1) & (d.group == "all faults")]
        sy = ["confidence alone", "C3", "REV3 (main)", f"{EV[ds][0]} alone", f"{EV[ds][1]} alone", "D (dropout model)"]
        st = [stats(d[d.system == s].auroc_harmful) for s in sy]
        x = np.arange(len(sy))
        ax.bar(x, [s[0] for s in st], yerr=[[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]], color="0.7",
               capsize=2, width=0.65)
        ax.bar(2, st[2][0], color="C0", width=0.65)
        ax.set_xticks(x, ["Conf.", "C3", "REV3", "TMC", "ECML", "Dropout"], rotation=35, ha="right"); ax.set_ylim(0.5, 1.0)
        ax.set_title(DSNAME[ds]); ax.set_ylabel("AUROC, harmful faults" if ds == "d1v2" else "")
    fig.tight_layout(); fig.savefig(o.base / "figs/F3_fault_detection.pdf"); fig.savefig(o.base / "figs/F3_fault_detection.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- main
# ----------------------------------------------------------------- T13-T16: the 14 Nov additions
# docs/plan_additions_2026-11-14.md. New functions only; the tables above are unchanged.
def additions_faults(o):
    """Addition 1: stuck sensor + saturation (faults2 bench), as T5 / T5b, plus harm by fault and level and the
    share of readings actually clipped."""
    fault_tables(o, prefix="faults2", logs={d: f"score_faults2_{d}.log" for d in DS}, tag="faults2",
                 names=("T13_faults_new", "T13b_faults_new_paired"),
                 titles=("Two more everyday faults, stuck sensor and saturation (frozen transfer, the new faults "
                         "pooled, revision 1)", "Stuck sensor and saturation: REV3 minus other systems, "
                         "harmful-fault AUROC, paired per unit"),
                 note="Stuck = the sensor freezes at 75 / 50 / 0 % of the clip until its end; saturation = readings "
                      "clipped at +/-8 / 4 / 2 g and +/-1000 / 500 / 250 deg/s; video: frozen frames and over-"
                      "exposure x1.5 / 2 / 3. Certificate fitted on validation lies only (frozen). The original "
                      "faults (T5) are not part of this pool.")
    rows = []
    for ds in DS:
        d = pd.read_csv(RES / f"faults2_{ds}_per_subject.csv")
        d = d[(d.revision == 1) & d.group.str.match(r"^f_(stuck|sat):L\d:all$")]
        for g, h in d.groupby("group", sort=False):
            f, lv = g.split(":")[0], int(g.split(":")[1][1])
            c, r = h[h.system == "confidence alone"], h[h.system == "REV3 (main)"]
            rows.append({"Dataset": DSNAME[ds], "Fault": {"f_stuck": "stuck", "f_sat": "saturation"}[f], "Level": lv,
                         "Answers changed %": fmt(r.flipped.mean(), k=100, d=1),
                         "Fault-clip acc. alone %": fmt(r.acc_fault.mean(), k=100, d=1),
                         "REV3 use/drop %": fmt(r.acc_policy_fault.mean(), k=100, d=1),
                         "AUROC harmful: confidence": fmt(c.auroc_harmful.mean()),
                         "AUROC harmful: REV3": fmt(r.auroc_harmful.mean()),
                         "REV3 names faulty sensor % (harmful)": fmt(r.naming_harmful.mean(), k=100, d=1)})
    o.table("T13c_faults_new_by_level", "Stuck sensor and saturation by fault and level, all sensors pooled "
            "(means over units, revision 1)", pd.DataFrame(rows))
    cs = pd.read_csv(RES / "faults2_clip_share.csv")
    g = cs.groupby(["dataset", "sensor", "level"], sort=False)[["clipped", "n_values", "clips_with_clipping",
                                                               "n_clips"]].sum().reset_index()
    o.table("T13d_saturation_clipped", "Saturation: share of readings actually clipped (test part, all folds)",
            pd.DataFrame({"Dataset": g.dataset.map(DSNAME), "Sensor": g.sensor.map(SNAME), "Level": g.level,
                          "Readings clipped %": [f"{100 * a / b:.3f}" for a, b in zip(g.clipped, g.n_values)],
                          "Clips with any clipping %": [f"{100 * a / b:.1f}" for a, b in
                                                        zip(g.clips_with_clipping, g.n_clips)]}),
            "Levels: 1 = +/-8 g, +/-1000 deg/s; 2 = +/-4 g, +/-500 deg/s; 3 = +/-2 g, +/-250 deg/s "
            "(scripts/faults2_clip_share.py).")


def additions_nowatch(o):
    """Addition 3: the dead watch as raw 0 on the 247 real no-watch clips (nowatch2 bench), as T6."""
    p = pd.read_csv(RES / "nowatch2_d2v_per_subject.csv"); p = p[p.revision == 1]
    big, logf = [2, 13, 16, 20], "score_nowatch2_d2v.log"
    logr = log_rows(log_block(logf, "SYSTEMS, revision 1"), 10)
    rows = []
    for s, g in p.groupby("system", sort=False):
        rate = lambda a, b: g[a].sum() / g[b].sum()
        v = {"acc_alone": 100 * rate("correct_alone_nw", "n_nw"), "acc_policy": 100 * rate("correct_policy_nw", "n_nw"),
             "ref": 100 * rate("correct_policy_ref", "n_ref"), "flagged": 100 * rate("n_flagged_nw", "n_nw"),
             "named": np.nan if s == "confidence alone" else 100 * rate("n_named_watch_flagged_nw", "n_flagged_nw"),
             "auroc": g.set_index("subject").loc[big].auroc_nw.mean()}
        lv = logr[s]
        for j, c in ((0, "acc_alone"), (1, "acc_policy"), (2, "ref"), (3, "flagged"), (5, "named")):
            check(f"nowatch2 {s} {c}", v[c], lv[j], 3)
        check(f"nowatch2 {s} auroc", v["auroc"], lv[7], 3)
        rows.append({"System": LABEL.get(s, s), "Acc. alone %": fmt(v["acc_alone"], d=1),
                     "Acc. after use/drop %": fmt(v["acc_policy"], d=1), "Flagged %": fmt(v["flagged"], d=1),
                     "Names the watch (flagged) %": fmt(v["named"], d=1),
                     "AUROC no-watch vs reference (4 people)": fmt(v["auroc"]),
                     "Acc. on reference clips %": fmt(v["ref"], d=1)})
    km = re.search(r"pooled\s*: doing nothing\s+([\d.]+) \| REV3 policy\s+([\d.]+) \| known missing\s+([\d.]+)",
                   (LOGS / logf).read_text())
    check("nowatch2 pooled REV3 policy (log line)", 100 * rate_all(p, "REV3 (main)", "correct_policy_nw", "n_nw"),
          float(km.group(2)), 1)
    rows.append({"System": "Backup without the watch, sensor known missing [reference]", "Acc. alone %": "--",
                 "Acc. after use/drop %": km.group(3), "Flagged %": "--", "Names the watch (flagged) %": "--",
                 "AUROC no-watch vs reference (4 people)": "--", "Acc. on reference clips %": "--"})
    per = p[p.system == "REV3 (main)"].set_index("subject")
    who = "; ".join(f"person {int(k)}: {100 * r.correct_policy_nw / r.n_nw:.1f}% vs doing nothing "
                    f"{100 * r.correct_alone_nw / r.n_nw:.1f}% ({int(r.n_nw // 3)} clips)" for k, r in per.iterrows())
    o.table("T14_nowatch_rawzero", "MMAct: the 247 real no-watch clips with the watch as raw 0 (addition 3; 6 people, "
            "3 seeds pooled; revision 1)", pd.DataFrame(rows),
            "Raw 0 on every watch channel, then the training normalisation (physically impossible while worn). "
            f"REV3 per person: {who}. The reference row is read from {logf}.")


def additions_imuoff(o):
    """Addition 2: k IMUs off during p = 25 / 50 / 75 % of the clip (imuoff2 bench); p = 100 % is T8."""
    d = pd.read_csv(RES / "imuoff2_d3_per_fold.csv"); d = d[(d.revision == 1) & (d.group != "clean")]
    pred = pd.read_csv(RES / "imuoff2_d3_predictions.csv")
    pred = pred[pred["item"].isin(["main", "supporting"])]
    rows, curve = [], []
    for (grp, k, pp), g in d.groupby(["group", "k", "p"], sort=False):
        rv = g[g.system == "REV3 (main)"].sort_values("fold")
        st = stats(100 * (rv.acc_policy.values - rv.acc_alone.values))
        pr = pred[(pred.group == grp) & (pred.k == k) & (pred.p == pp)]
        if len(pr) != 1:
            raise SystemExit(f"imuoff2 {grp} k={k} p={pp}: {len(pr)} rows in the predictions file")
        check(f"imuoff2 {grp} k={k} p={pp} REV3 minus nothing (points)", st[0], pr["mean"].iloc[0], 9)
        rows.append({"Group": grp, "IMUs off": int(k), "Off for % of the clip": int(pp),
                     "Doing nothing %": fmt(rv.acc_alone.mean(), k=100, d=1),
                     "REV3 (ours, main) %": fmt(rv.acc_policy.mean(), k=100, d=1),
                     "REV3 minus nothing (points)": fmt(*st[:3], d=2, sign=True), "Folds better": f"{st[3]}/{st[4]}",
                     "REV3 AUROC": fmt(rv.auroc.mean())})
        curve.append((grp, int(k), int(pp), *st[:3]))
    o.table("T15_imuoff_partial", "WearGait-PD: k IMUs switched off during part of the clip (addition 2; mean over "
            "10 folds; revision 1)", pd.DataFrame(rows), "p = 100 % (whole clip) is table T8.")
    return pd.DataFrame(curve, columns=["group", "k", "p", "mean", "lo", "hi"])


def additions_predictions(o):
    """Every prediction of the 14 Nov additions with its verdict, read from the scorers' prediction files, plus
    P3 of the v1 WearGait-PD ablation, which is decided by the numbers in T11."""
    v = pd.read_csv(RES / "v1_d3_auroc_summary.csv"); v = v[v.task == "LIE"].set_index("score")
    au = lambda k: float(v.loc[k, "auroc"].split()[0])
    signs = ["S1 stability drop", "S1 stability flip", "S2 hide-one flips", "S2 hide-one shift"]
    best = au("X lowest expert support")
    rows = [("v1 signs, WearGait-PD", "P3", "CONFIRMED" if all(au(k) < best for k in signs) else "NOT CONFIRMED",
             f"v1 signs alone {min(au(k) for k in signs):.3f}-{max(au(k) for k in signs):.3f} vs lowest expert "
             f"support {best:.3f} (lie AUROC)")]
    for f, label in (("nowatch2_d2v_predictions.csv", "raw-zero watch, MMAct"),
                     ("imuoff2_d3_predictions.csv", "partial IMU-off, WearGait-PD")):
        q = pd.read_csv(RES / f)
        q = q[q.verdict.notna()] if "verdict" in q else q
        rows += [(label, r["item"], r["verdict"], r["value"]) for _, r in q.iterrows()]
    for ds in DS:
        q = pd.read_csv(RES / f"faults2_{ds}_predictions.csv")
        rows += [(f"stuck + saturation, {DSNAME[ds]}", r["item"], r["verdict"], r["value"]) for _, r in q.iterrows()]
    o.table("T16_predictions_additions", "Predictions written before the runs (13-14 Nov) and their verdicts",
            pd.DataFrame(rows, columns=["Experiment", "Prediction", "Verdict", "Result"]),
            "Plans: docs/plan_v1_d3_2026-11-13.md and docs/plan_additions_2026-11-14.md. A prediction "
            "that failed is reported as failed; nothing was re-tuned after a result.")


def figure_partial(o, c2, c1):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    whole = c1[c1.group == "legs"].assign(p=100)
    allp = pd.concat([c2[c2.group == "legs"], whole])
    for k, mk in ((8, "o"), (6, "s"), (4, "^"), (2, "v")):
        c = allp[allp.k == k].sort_values("p")
        ax.errorbar(c.p, c["mean"], yerr=[c["mean"] - c.lo, c.hi - c["mean"]], fmt=mk + "-", ms=3.5, lw=1,
                    capsize=2, label=f"{k} of 8 IMUs")
    ax.axhline(0, color="0.5", lw=0.6, ls="--")
    ax.set_xticks([25, 50, 75, 100]); ax.set_xlabel("Legs: IMUs off for % of the clip")
    ax.set_ylabel("REV3 minus doing nothing\n(accuracy points)"); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(o.base / "figs/F4_imuoff_partial.pdf"); fig.savefig(o.base / "figs/F4_imuoff_partial.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper/journal"))
    ap.add_argument("--skip-models", action="store_true", help="skip T1 (needs runs/*/preds_test.parquet)")
    a = ap.parse_args()
    o = Out(a.out)
    if not a.skip_models:
        t_models(o)
    lie_tables(o)
    fault_tables(o)
    curve = missing_tables(o)
    other_tables(o)
    figures(o, curve)
    additions_faults(o)
    additions_nowatch(o)
    curve2 = additions_imuoff(o)
    additions_predictions(o)
    figure_partial(o, curve2, curve)
    bad = [c for c in CHECKS if not c[4]]
    lines = [f"{'OK ' if ok else 'BAD'} {w}: built {x:.6f} | scorer log {y:.6f} (tolerance {tol:.1e})"
             for w, x, y, tol, ok in CHECKS]
    (Path(a.out) / "checks.txt").write_text("\n".join(lines) + "\n")
    o.md += ["## Checks", "", f"{len(CHECKS)} numbers compared with the scorers' own logs; {len(bad)} disagree.", ""]
    (Path(a.out) / "NUMBERS.md").write_text("\n".join(o.md) + "\n")
    print(f"tables: {len(list((Path(a.out) / 'tables').glob('*.tex')))} | figures: "
          f"{len(list((Path(a.out) / 'figs').glob('*.pdf')))} | checks: {len(CHECKS)} compared, {len(bad)} disagree"
          f" (of which {sum(c[0].startswith('T1 ') for c in CHECKS)} model-accuracy checks against the training logs)")
    for c in bad:
        print(f"  BAD {c[0]}: built {c[1]:.6f} vs log {c[2]:.6f}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
