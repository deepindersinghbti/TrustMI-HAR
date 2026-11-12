"""Fault plan, step 5 (docs/plan_faults_2026-11-11.md, section 6): score the certificate on REAL sensor faults.

  python scripts/cert_fault_score.py --dataset d1v2
  python scripts/cert_fault_score.py --dataset d2v
  ... --tag f0_s0 --novideo     the queue F smoke-run files (code check only; never overwrites the full-run CSV)

TRANSFER, the main test: every system is fitted on each fold's VALIDATION person with the LIE bench only (swaps,
same-action swaps, harmless changes, clean), exactly as for every lie number so far: weights of the score, the
naming rule, the use / drop threshold (revisions 1 and 2) and the abstain score. It is then applied UNCHANGED to
the TEST person's FAULT bench (clean + drift / gap / shift at 3 levels). Nothing is refitted, nothing retrained.

Systems (all from existing scorers, imported, not edited):
  confidence alone     1 - msp of the fused model; no naming, no drop (use / abstain only)
  C3                   1 - msp, witness JS, lowest witness support; odd one out; backups (configs/backups.json)
  REV3 (main)          C3 + MI; odd one out or lowest MI share (chosen on validation); backups
  TMC, ECML            their own signals, rules and drop (their fusion without the named sensor)
  D (dropout model)    REV3 judging the 50% sensor-dropout model; drop = that model with the sensor hidden
  REV3 known faults    as REV3, but fitted on the validation person with the fault clips added to the positives
                       (the naming rule and threshold see them too). Reported NEXT TO, never instead of, REV3.

Per test person, per fault condition and per pooled group (fault x level over sensors, fault over levels, all):
  harm        accuracy of the system's own answer on the fault clips vs the same clips clean, share flipped
  detection   AUROC of the score: fault clips vs the person's clean clips; and HARMFUL fault clips only
              (the system's own answer changed because of the fault; >= 5 such clips, else not scored)
  flagged     share of fault clips at or above the drop threshold, and of clean clips (false alarms)
  naming      faulty sensor named (all fault clips, harmful ones)
  recovery    accuracy on the fault clips: answer alone vs the use / drop policy; same on clean clips;
              "answer the safest 80%" on the stream clean + those fault clips
CHECKS before any number: (1) the fault bench's clean rows equal the lie bench's clean rows (same models, same
inputs); (2) fitting frozen and applying to the LIE test rows reproduces per_subject (the scorer behind every lie
number) exactly, for every system and revision.
Writes results/certificate/faults_<dataset>[_<tag>]_per_subject.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cert_policy as CP
import cert_mi_score as MS
import cert_dropout_score as DSC
from cert_evidential_score import (per_subject, name, ev_feats, ev_drop, EV_RULES, ci, RES, is_lie, EVFILE)
from cert_score import auroc, fit_logreg, MIN_CLASS, unit

KEYS = MS.KEYS
FULL = "all_s0-1-2"
SYS_CONF, SYS_C3, SYS_MAIN, SYS_D, SYS_KNOWN = ("confidence alone", "C3", "REV3 (main)", "D (dropout model)",
                                                "REV3 known faults")
REV3_SIG, REV3_RULES = MS.SYSTEMS[MS.MAIN]
CHECK_COLS = ["lie_auroc", "naming", "acc_policy_lie", "acc_policy_normal", "all_cov80_cert_policy"]


def is_fault(d):
    return d.condition.str.startswith("f_").to_numpy()


# ---------------------------------------------------------------- a system fitted once, applied anywhere
class Frozen:
    """Exactly per_subject's fitting (cert_evidential_score), split into fit (validation) and apply (any rows)."""

    def __init__(self, va, ds, feats, rules, drop, rev, pos=None):
        self.ds, self.feats, self.rules, self.drop, self.rev = ds, feats, rules, drop, rev
        lv = is_lie(va) if pos is None else pos
        Xv = feats(va, ds)
        self.mu, self.sd = Xv.mean(0), Xv.std(0) + 1e-9
        self.w = fit_logreg((Xv - self.mu) / self.sd, np.asarray(lv, float))
        Dv = self._lin(Xv, self.w)
        cv, yv, fv = (va.condition == "clean").to_numpy(), va.y.to_numpy(), va.pred.to_numpy()
        if rules:
            L = va[lv]
            acc = {q: np.mean(name(L, ds, rules, q) == L.attacked.to_numpy()) for q in rules}
            self.rule = max(acc, key=acc.get)
            bv = drop(va, ds, name(va, ds, rules, self.rule))
            if rev == 1:
                cand = np.r_[np.inf, np.quantile(Dv, np.linspace(1, 0, 101))]
                self.t = cand[int(np.argmax([np.mean(np.where(Dv >= c, bv, fv) == yv) for c in cand]))]
            else:
                self.t = np.nextafter(np.quantile(Dv[cv], 1 - CP.BUDGET), np.inf)
            final_v = np.where(Dv >= self.t, bv, fv)
        else:
            self.rule, self.t, final_v = "", np.inf, fv
        self.w2 = fit_logreg((Xv - self.mu) / self.sd, (final_v != yv).astype(float)) if rev == 2 else None

    def _lin(self, X, w):
        return np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ w

    def apply(self, te):
        X = self.feats(te, self.ds)
        D = self._lin(X, self.w)
        f = te.pred.to_numpy()
        if self.rules:
            named = name(te, self.ds, self.rules, self.rule)
            final = np.where(D >= self.t, self.drop(te, self.ds, named), f)
        else:
            named, final = np.full(len(te), ""), f
        A = D if self.rev == 1 else self._lin(X, self.w2)
        return D, named, final, A


# ---------------------------------------------------------------- data
def merge(df, other, cols, what):
    return DSC.merge_cols(df, other, cols, what)


def fault_frame(ds, tag, fs):
    ex = pd.read_parquet(RES / f"experts_{ds}_{tag}{fs}.parquet")
    ex = ex.assign(pred=ex.pred_fused)
    mi = pd.read_parquet(RES / f"mi_{ds}_own_{tag}{fs}.parquet")
    return merge(ex, mi, [c for c in mi.columns if c.startswith(("pmi_", "mi_", "s4"))], f"{ds} fault MI")


def with_dropout(df, f, ds):
    sd = pd.read_parquet(f)
    own = [c for c in sd.columns if c.startswith("sd_")]
    sd = sd.rename(columns={c: f"sd50_{c[3:]}" for c in own})
    return DSC.as_main(merge(df, sd, [f"sd50_{c[3:]}" for c in own], f.name), ds, "sd50")


def check_clean(lie, flt, cols, label):
    """The two benches must agree exactly on the clean rows (same models, same inputs)."""
    a = lie[lie.condition == "clean"].set_index(KEYS)
    b = flt[flt.condition == "clean"].set_index(KEYS)
    if len(b) == 0 or not b.index.isin(a.index).all():
        raise SystemExit(f"{label}: fault-bench clean rows missing from the lie bench: stop and paste this")
    a = a.loc[b.index]
    worst = 0.0
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        worst = max(worst, float(np.max(np.abs(x.astype(float) - y.astype(float)))))
    if worst > 1e-6:
        raise SystemExit(f"{label}: clean rows differ between the two benches by {worst:.2e}: stop and paste this")
    print(f"check ({label}): clean rows identical in the lie and fault benches ({len(b)} rows, largest diff {worst:.1e})")


# ---------------------------------------------------------------- metrics
def groups(te):
    """(group name, fault, level, sensor, mask) for every fault condition and the pooled groups."""
    c = te.condition.to_numpy()
    parts = pd.Series(c).str.split(":", expand=True)
    out = []
    fl = parts[0].to_numpy()
    for cond in sorted(set(c) - {"clean"}):
        f, lv, s = cond.split(":")
        out.append((cond, f, lv, s, c == cond))
    if len(out):
        lvl = parts[1].to_numpy()
        for f in sorted(set(fl) - {"clean"}):
            for lv in ("L1", "L2", "L3"):
                k = (fl == f) & (lvl == lv)
                if k.any():
                    out.append((f"{f}:{lv}:all", f, lv, "all", k))
            out.append((f"{f}:all:all", f, "all", "all", fl == f))
        out.append(("all faults", "all", "all", "all", fl != "clean"))
    return out


def fault_rows(te, D, named, final, A, t, system, rev):
    """One dict per group for one test person."""
    y, pred, cond = te.y.to_numpy(), te.pred.to_numpy(), te.condition.to_numpy()
    att, msp = te.attacked.to_numpy(), te.msp.to_numpy()
    k0 = cond == "clean"
    clean_pred = pd.Series(pred[k0], index=pd.MultiIndex.from_arrays([te.seed.to_numpy()[k0], te["clip"].to_numpy()[k0]]))
    ref = clean_pred.reindex(pd.MultiIndex.from_arrays([te.seed.to_numpy(), te["clip"].to_numpy()])).to_numpy()
    flip = pred != ref
    rows = []
    for gname, f, lv, s, k in groups(te):
        h = k & flip
        both, both_h = k | k0, h | k0
        r = dict(fold=int(te.fold.iloc[0]), subject=unit(te), system=system, revision=rev, group=gname,
                 fault=f, level=lv, sensor=s, n=int(k.sum()), n_harmful=int(h.sum()),
                 acc_clean=np.mean(pred[k0] == y[k0]), acc_fault=np.mean(pred[k] == y[k]), flipped=np.mean(flip[k]),
                 auroc=auroc(D[both], k[both]),
                 auroc_harmful=auroc(D[both_h], h[both_h], MIN_CLASS) if h.sum() >= MIN_CLASS else np.nan,
                 flagged=np.mean(D[k] >= t), false_alarm=np.mean(D[k0] >= t),
                 acc_policy_fault=np.mean(final[k] == y[k]), acc_policy_clean=np.mean(final[k0] == y[k0]),
                 cov80_policy=CP.selective(A[both], final[both], y[both], 0.8),
                 cov80_conf=CP.selective(1 - msp[both], pred[both], y[both], 0.8))
        if (named != "").any():
            r["naming"] = np.mean(named[k] == att[k])
            r["naming_harmful"] = np.mean(named[h] == att[h]) if h.sum() >= MIN_CLASS else np.nan
        rows.append(r)
    return rows


def check_frozen_on_lies(fz, te, ref, label):
    """Frozen fit applied to the LIE test rows must equal per_subject's row for this person."""
    D, named, final, A = fz.apply(te)
    lt, yt = is_lie(te), te.y.to_numpy()
    got = dict(lie_auroc=auroc(D, lt), acc_policy_lie=np.mean(final[CP.group(te.condition) == "lie"] == yt[CP.group(te.condition) == "lie"]),
               acc_policy_normal=np.mean(final[(te.condition == "clean").to_numpy()] == yt[(te.condition == "clean").to_numpy()]),
               all_cov80_cert_policy=CP.selective(A, final, yt, 0.8),
               naming=np.mean(named[lt] == te.attacked.to_numpy()[lt]) if fz.rules else np.nan)
    for c in CHECK_COLS:
        a, b = got[c], ref[c]
        if not ((np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9):
            raise SystemExit(f"{label}: frozen fit gives {c} = {a} on the lie test rows, per_subject {b}: stop and paste this")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=CP.JOURNAL, required=True)
    ap.add_argument("--tag", default=FULL, help=f"{FULL} = the full run; f0_s0 = smoke-run files (code check only)")
    ap.add_argument("--novideo", action="store_true", help="read the _faults-novideo smoke files")
    args = ap.parse_args(); ds = args.dataset
    fs = "_faults-novideo" if args.novideo else "_faults"
    MS.journal_files(ds)                                  # lie files: always the full run (CP.TAG = all_s0-1-2)
    pd.set_option("display.width", 220)

    # ---- lie bench (fitting side) and fault bench (applying side), same folds and seeds
    flt = fault_frame(ds, args.tag, fs)
    folds, seeds = sorted(flt.fold.unique()), sorted(flt.seed.unique())
    keep = lambda d: d[d.fold.isin(folds) & d.seed.isin(seeds)]
    lie = keep(MS.add_mi(CP.load(ds), ds))
    lie = lie[lie.available]
    check_clean(lie, flt, ["pred", "msp", "exp_js", "s4"] + [f"exp_support_{m}" for m in CP.EXP[ds]], f"{ds} B4")
    ev_l = keep(pd.read_parquet(RES / f"{EVFILE[ds]}.parquet"))
    ev_f = pd.read_parquet(RES / f"evidential_{ds}_{args.tag}{fs}.parquet")
    ev_l = ev_l[ev_l.available]
    for m in MS.MAIN_EV[ds]:
        check_clean(ev_l[ev_l.method == m], ev_f[ev_f.method == m], ["pred", "msp", "u", "conflict"], f"{ds} {m}")
    d_l = with_dropout(lie, RES / f"dropout_{ds}_sd50_{FULL}.parquet", ds)
    d_f = with_dropout(flt, RES / f"dropout_{ds}_sd50_{args.tag}{fs}.parquet", ds)
    check_clean(d_l, d_f, ["pred", "msp"], f"{ds} dropout sd50")

    rev3 = MS.signals(REV3_SIG)
    systems = {   # name -> (lie df, fault df, features, naming rules, drop)
        SYS_CONF: (lie, flt, MS.signals(["s1"]), None, None),
        SYS_C3: (lie, flt, MS.signals(["s1", "s2", "s3"]), {"odd one out": MS.ODD}, CP.backup_answer),
        SYS_MAIN: (lie, flt, rev3, REV3_RULES, CP.backup_answer),
        **{f"{m.upper()} alone": (ev_l[ev_l.method == m], ev_f[ev_f.method == m], ev_feats, EV_RULES, ev_drop)
           for m in MS.MAIN_EV[ds]},
        SYS_D: (d_l, d_f, rev3, REV3_RULES, DSC.hide_answer("sd50_hide_pred")),
    }

    rows = []
    for name_, (L, F, feats, rules, drop) in systems.items():
        for rev in (1, 2):
            ref = per_subject(L, ds, feats, rules or {}, drop or (lambda d, ds_, n: d.pred.to_numpy()), rev) \
                if rules else None
            for fold in folds:
                va, te_l = L[(L.fold == fold) & (L.part == "val")], L[(L.fold == fold) & (L.part == "test")]
                te_f = F[(F.fold == fold) & (F.part == "test")]
                fz = Frozen(va, ds, feats, rules, drop, rev)
                if ref is not None:
                    check_frozen_on_lies(fz, te_l, ref.loc[unit(te_l)], f"{ds} {name_} rev {rev} fold {fold}")
                rows += fault_rows(te_f, *fz.apply(te_f), fz.t, name_, rev)
        print(f"scored: {name_}" + (" (frozen fit reproduces the lie numbers exactly)" if rules else ""))

    # ---- the labelled extra row: REV3 fitted with the faults among the validation positives
    for rev in (1, 2):
        for fold in folds:
            va = pd.concat([lie[(lie.fold == fold) & (lie.part == "val")],
                            flt[(flt.fold == fold) & (flt.part == "val") & (flt.condition != "clean")]], ignore_index=True)
            pos = is_lie(va) | is_fault(va)
            fz = Frozen(va, ds, rev3, REV3_RULES, CP.backup_answer, rev, pos=pos)
            te_f = flt[(flt.fold == fold) & (flt.part == "test")]
            rows += fault_rows(te_f, *fz.apply(te_f), fz.t, SYS_KNOWN, rev)
    print(f"scored: {SYS_KNOWN} (labelled extra row)")

    res = pd.DataFrame(rows)
    report(res, ds)
    out = RES / f"faults_{ds if args.tag == FULL and not args.novideo else f'{ds}_{args.tag}{fs}'}_per_subject.csv"
    res.to_csv(out, index=False)
    print(f"\nsaved: {out.relative_to(ROOT)}")


def mean_table(res, rev, cols, groups_, systems):
    q = res[(res.revision == rev) & res.group.isin(groups_) & res.system.isin(systems)]
    t = q.groupby(["group", "system"], sort=False)[cols].mean()
    return t.reindex(pd.MultiIndex.from_product([groups_, systems], names=["group", "system"])).dropna(how="all")


def report(res, ds):
    systems = list(dict.fromkeys(res.system))
    pooled = [g for g in dict.fromkeys(res.group) if g.endswith(":all") or g == "all faults"]
    single = [g for g in dict.fromkeys(res.group) if g not in pooled]
    p = lambda t: print((t * 1).round(3).to_string())
    print(f"\n{ds}: HARM, fused model alone (mean over test people; accuracy and share of answers flipped)")
    h = res[(res.system == SYS_C3) & (res.revision == 1) & res.group.isin(single)]
    p(h.groupby("group", sort=False)[["n", "acc_clean", "acc_fault", "flipped", "n_harmful"]].mean())
    print(f"\n{ds}: DETECTION, AUROC fault clips vs clean clips (all / harmful only), pooled groups")
    t = mean_table(res, 1, ["auroc", "auroc_harmful"], pooled, systems)
    p(t.unstack("system")["auroc_harmful"].reindex(columns=systems))
    print("  (all fault clips:)")
    p(t.unstack("system")["auroc"].reindex(columns=systems))
    print(f"\n{ds}: NAMING the faulty sensor, % (harmful fault clips), pooled groups")
    named = [s for s in systems if s != SYS_CONF]
    p(100 * mean_table(res, 1, ["naming_harmful"], pooled, named).unstack("system")["naming_harmful"].reindex(columns=named))
    for rev in (1, 2):
        print(f"\n{ds}: RECOVERY, revision {rev}" + (" (post hoc, labelled)" if rev == 2 else "")
              + ": accuracy % on fault clips (alone -> policy), on clean clips, flagged %, false alarms %")
        t = mean_table(res, rev, ["acc_fault", "acc_policy_fault", "acc_policy_clean", "flagged", "false_alarm"],
                       ["all faults"] + [g for g in pooled if g.endswith(":all:all")], named)
        p(100 * t)
    print(f"\n{ds}: PAIRED, REV3 (main) minus each other system, per test person, revision 1, all faults pooled")
    a = res[(res.system == SYS_MAIN) & (res.revision == 1) & (res.group == "all faults")].set_index("subject")
    for s in [x for x in systems if x != SYS_MAIN]:
        b = res[(res.system == s) & (res.revision == 1) & (res.group == "all faults")].set_index("subject").loc[a.index]
        line = (f"  minus {s:20s} AUROC harmful {ci(a.auroc_harmful - b.auroc_harmful, 1, 3)} | "
                f"AUROC all {ci(a.auroc - b.auroc, 1, 3)} | fault-clip acc {ci(a.acc_policy_fault - b.acc_policy_fault, 100)} | "
                f"clean acc {ci(a.acc_policy_clean - b.acc_policy_clean, 100)}")
        if "naming_harmful" in b and b.naming_harmful.notna().any():
            line += f" | naming harmful {ci(a.naming_harmful - b.naming_harmful, 100)}"
        print(line)


if __name__ == "__main__":
    main()
