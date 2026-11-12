"""WearGait-PD real-dropout test (docs/plan_weargait_2026-11-11.md, 2.6 and implementation notes 8):
score every system on the clips where sensor groups REALLY lost samples.

  python scripts/cert_realdrop_score.py                   full run (queue I files, tag all_s0-1-2)
  python scripts/cert_realdrop_score.py --tag f0_s0       smoke-run files (code check only; never overwrites)

FROZEN, as in the fault and no-watch tests: every system is fitted on each fold's VALIDATION people with the
LIE bench only (cert_fault_score.Frozen) and applied unchanged to the fold's test people:
  rd    rd_fill:<groups>  real-dropout clips; missing samples at the training mean. 'single' = exactly one group
                          lost samples (the main set); 'multi' = two or three groups (reported separately)
  ref   clean             all clean clips of the fold's test people
  rf    r_fill:<group>    the same clean clips with that group set to its training mean (simulated removal)
Answers without a certificate: doing nothing (full model); known missing (the backup chosen for the missing group,
configs/backups.json; for two missing groups the witness of the remaining group; none for three); M (the missing
group(s) flagged absent in the full model); the dropout model with the missing group hidden (single only).
MAIN RESULT, named before any result: REV3's use / drop accuracy (revision 1) on the single-group real-dropout
clips, pooled over clips, against doing nothing and against known missing; also per fold.
Detection AUROC (rd single vs ref) per fold with at least 20 single-group real-dropout clips, mean over those
folds; never pooled across folds (different fits). Naming = the named group is the group that lost samples.
CHECKS before any number: clean rows identical to the lie bench; each frozen fit reproduces the lie numbers.
Writes results/certificate/realdrop_d3[_<tag>]_per_fold.csv and realdrop_d3[_<tag>]_rows.parquet
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
from cert_evidential_score import per_subject, ev_feats, ev_drop, EV_RULES, RES, EVFILE
from cert_score import auroc, unit
from cert_fault_score import (Frozen, check_clean, check_frozen_on_lies, with_dropout, merge, FULL, KEYS,
                              SYS_CONF, SYS_C3, SYS_MAIN, SYS_D, REV3_SIG, REV3_RULES)

DS_ = "d3"
SENS = ("arms", "trunk", "legs")
MIN_FOLD = 20                                 # single-group real-dropout clips a fold needs for its own AUROC


def bench(tag):
    ex = pd.read_parquet(RES / f"experts_{DS_}_{tag}_realdrop.parquet")
    ex = ex.assign(pred=ex.pred_fused)
    mi = pd.read_parquet(RES / f"mi_{DS_}_own_{tag}_realdrop.parquet")
    b = merge(ex, mi, [c for c in mi.columns if c.startswith(("pmi_", "mi_", "s4"))], f"{DS_} real-dropout MI")
    return b[b.available].reset_index(drop=True)          # each real-dropout clip only under its own pattern


def sets_of(b):
    c = b.condition.to_numpy()
    out = np.full(len(b), "", dtype=object)
    out[c == "clean"] = "ref"
    out[np.char.startswith(c.astype(str), "r_fill:")] = "rf"
    rd = np.char.startswith(c.astype(str), "rd_fill:")
    n_groups = np.array([len(a.split(",")) if a else 0 for a in b.attacked.astype(str)])
    out[rd & (n_groups == 1)] = "rd1"
    out[rd & (n_groups > 1)] = "rdm"
    return out


def known_answer(b):
    """Known missing: the chosen backup for one missing group; the remaining group's witness for two."""
    out = np.full(len(b), -1)
    att = b.attacked.astype(str).to_numpy()
    for g in SENS:
        k = att == g
        out[k] = b[CP.backup_col(CP.BACKUPS[DS_][g])].to_numpy()[k]
    for left in SENS:
        k = np.array([set(a.split(",")) == set(SENS) - {left} for a in att])
        out[k] = b[f"exp_pred_{left}"].to_numpy()[k]
    return out


def dropout_hidden(b):
    out = np.full(len(b), -1)
    att = b.attacked.astype(str).to_numpy()
    for g in SENS:
        k = att == g
        out[k] = b[f"sd50_hide_pred_{g}"].to_numpy()[k]
    return out


def rows_for(te, D, named, final, A, t, system, rev, sets):
    k = sets != ""
    return pd.DataFrame(dict(system=system, revision=rev, fold=te.fold.to_numpy()[k], unit=unit(te),
                             seed=te.seed.to_numpy()[k], clip=te["clip"].to_numpy()[k], set=sets[k],
                             group=te.attacked.astype(str).to_numpy()[k], y=te.y.to_numpy()[k],
                             pred=te.pred.to_numpy()[k], final=final[k], score=D[k], abstain_score=A[k],
                             flagged=D[k] >= t, named=named[k], msp=te.msp.to_numpy()[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=FULL, help=f"{FULL} = full run; e.g. f0_s0 = smoke-run files (code check)")
    args = ap.parse_args(); ds = DS_
    if ds not in CP.BACKUPS:
        raise SystemExit("configs/backups.json has no d3 entry: run choose_backups_journal.py --datasets d3 first")
    MS.journal_files(ds)
    pd.set_option("display.width", 220)

    b = bench(args.tag)
    folds, seeds = sorted(b.fold.unique()), sorted(b.seed.unique())
    keep = lambda d: d[d.fold.isin(folds) & d.seed.isin(seeds)]
    lie = keep(MS.add_mi(CP.load(ds), ds))
    lie = lie[lie.available]
    check_clean(lie, b[b.part == "test"], ["pred", "msp", "exp_js", "s4"] + [f"exp_support_{m}" for m in CP.EXP[ds]],
                f"{ds} B4")
    ev_l = keep(pd.read_parquet(RES / f"{EVFILE[ds]}.parquet")); ev_l = ev_l[ev_l.available]
    ev_b = pd.read_parquet(RES / f"evidential_{ds}_{args.tag}_realdrop.parquet"); ev_b = ev_b[ev_b.available]
    ev_b = ev_b.merge(b[KEYS + ["attacked"]].rename(columns={"attacked": "att_b"}), on=KEYS, how="inner",
                      validate="many_to_one")
    for m in MS.MAIN_EV[ds]:
        check_clean(ev_l[ev_l.method == m], ev_b[(ev_b.method == m) & (ev_b.part == "test")],
                    ["pred", "msp", "u", "conflict"], f"{ds} {m}")
    d_l = with_dropout(lie, RES / f"dropout_{ds}_sd50_{FULL}.parquet", ds)
    raw = b.copy()
    d_b = with_dropout(b, RES / f"dropout_{ds}_sd50_{args.tag}_realdrop.parquet", ds)
    check_clean(d_l, d_b[d_b.part == "test"], ["pred", "msp"], f"{ds} dropout sd50")
    raw = raw.merge(d_b[KEYS + ["sd50_pred"] + [f"sd50_hide_pred_{g}" for g in SENS]], on=KEYS, validate="one_to_one")

    rev3 = MS.signals(REV3_SIG)
    systems = {
        SYS_CONF: (lie, b, MS.signals(["s1"]), None, None),
        SYS_C3: (lie, b, MS.signals(["s1", "s2", "s3"]), {"odd one out": MS.ODD}, CP.backup_answer),
        SYS_MAIN: (lie, b, rev3, REV3_RULES, CP.backup_answer),
        **{f"{m.upper()} alone": (ev_l[ev_l.method == m], ev_b[ev_b.method == m], ev_feats, EV_RULES, ev_drop)
           for m in MS.MAIN_EV[ds]},
        SYS_D: (d_l, d_b, rev3, REV3_RULES, DSC.hide_answer("sd50_hide_pred")),
    }
    rows = []
    for name_, (L, B, feats, rules, drop) in systems.items():
        sets = sets_of(B)
        for rev in (1, 2):
            ref = per_subject(L, ds, feats, rules or {}, drop or (lambda d, ds_, n: d.pred.to_numpy()), rev) \
                if rules else None
            for fold in folds:
                va, te_l = L[(L.fold == fold) & (L.part == "val")], L[(L.fold == fold) & (L.part == "test")]
                kb = (B.fold == fold).to_numpy()
                fz = Frozen(va, ds, feats, rules, drop, rev)
                if ref is not None:
                    check_frozen_on_lies(fz, te_l, ref.loc[unit(te_l)], f"{ds} {name_} rev {rev} fold {fold}")
                rows.append(rows_for(B[kb], *fz.apply(B[kb]), fz.t, name_, rev, sets[kb]))
        print(f"scored: {name_}" + (" (frozen fit reproduces the lie numbers exactly)" if rules else ""))
    R = pd.concat(rows, ignore_index=True)
    key = ["fold", "seed", "clip", "set"]
    base = R[(R.system == SYS_MAIN) & (R.revision == 1)][key].sort_values(key).reset_index(drop=True)
    for (s_, r_), g in R.groupby(["system", "revision"]):
        if not g[key].sort_values(key).reset_index(drop=True).equals(base):
            raise SystemExit(f"{s_} rev {r_}: different rows from REV3: stop and paste this")
    print("check: every system scored on the same rows " + str(base.set.value_counts().to_dict()))

    sets = sets_of(raw)
    ans = pd.DataFrame(dict(fold=raw.fold, seed=raw.seed, set=sets, group=raw.attacked.astype(str), y=raw.y,
                            nothing=raw.pred, known=known_answer(raw), flagged_absent=raw.mask_missing_pred,
                            dropout_hidden=dropout_hidden(raw), dropout_alone=raw.sd50_pred))
    report(R, ans)
    sfx = "" if args.tag == FULL else f"_{args.tag}"
    R.to_parquet(RES / f"realdrop_{ds}{sfx}_rows.parquet", index=False)
    per_fold(R).to_csv(RES / f"realdrop_{ds}{sfx}_per_fold.csv", index=False)
    print(f"\nsaved: results/certificate/realdrop_{ds}{sfx}_per_fold.csv and _rows.parquet")


def per_fold(R):
    out = []
    for (s_, r_, f), g in R.groupby(["system", "revision", "fold"]):
        rd, ref = g[g.set == "rd1"], g[g.set == "ref"]
        both = pd.concat([rd, ref])
        n_rd = rd.clip.nunique()
        out.append(dict(system=s_, revision=r_, fold=f, n_rd_clips=n_rd,
                        auroc=auroc(both.score.to_numpy(), (both.set == "rd1").to_numpy()) if n_rd >= MIN_FOLD else np.nan,
                        acc_alone_rd=(rd.pred == rd.y).mean(), acc_policy_rd=(rd.final == rd.y).mean(),
                        acc_policy_ref=(ref.final == ref.y).mean(), flagged_rd=rd.flagged.mean(),
                        false_alarm_ref=ref.flagged.mean(),
                        naming_rd=(rd.named == rd.group).mean(),
                        naming_rd_flagged=(rd[rd.flagged].named == rd[rd.flagged].group).mean()
                        if rd.flagged.any() else np.nan))
    return pd.DataFrame(out)


def report(R, ans):
    pct = lambda x: f"{100 * x:5.1f}"
    print("\nCLIPS: " + ", ".join(f"{k} {v}" for k, v in
                                  ans.groupby("set").size().drop("", errors="ignore").to_dict().items()) +
          " rows (clips x seeds)")
    print("\nANSWERS WITHOUT A CERTIFICATE, accuracy % (pooled over clips and seeds)")
    tab = {}
    for s, lab in (("rd1", "real dropout, one group"), ("rdm", "real dropout, 2-3 groups"), ("ref", "clean clips")):
        a = ans[ans.set == s]
        tab[lab] = {c: (100 * (a[c][a[c] >= 0] == a.y[a[c] >= 0]).mean() if (a[c] >= 0).any() else np.nan)
                    for c in ("nothing", "known", "flagged_absent", "dropout_alone", "dropout_hidden")}
    print(pd.DataFrame(tab).T.round(1).to_string())
    a = ans[ans.set == "rd1"]
    print("  one group, by group: " + " | ".join(
        f"{g}: n={len(a[a.group == g])} nothing {pct((a[a.group == g].nothing == a[a.group == g].y).mean())}"
        f" known {pct((a[a.group == g].known == a[a.group == g].y).mean())}" for g in SENS if (a.group == g).any()))

    m = R[(R.system == SYS_MAIN) & (R.revision == 1)]
    rd = m[m.set == "rd1"]
    print("\nMAIN RESULT (named in advance): REV3 (main), revision 1, accuracy % on single-group real-dropout clips")
    k = ans[ans.set == "rd1"]
    print(f"  pooled: doing nothing {pct((rd.pred == rd.y).mean())} | REV3 policy {pct((rd.final == rd.y).mean())}"
          f" | known missing {pct((k.known == k.y).mean())}  ({rd.clip.nunique()} clips x {rd.seed.nunique()} seeds)")
    for f, g in rd.groupby("fold"):
        kk = k[k.fold == f]
        print(f"  fold {f}: n={g.clip.nunique():4d} | nothing {pct((g.pred == g.y).mean())} | REV3 "
              f"{pct((g.final == g.y).mean())} | known {pct((kk.known == kk.y).mean())}")

    pf = per_fold(R)
    for rev in (1, 2):
        print(f"\nSYSTEMS, revision {rev}" + (" (post hoc, labelled)" if rev == 2 else "")
              + f": pooled over clips; AUROC = mean over folds with >= {MIN_FOLD} single-group clips")
        out = {}
        for s_, g in R[R.revision == rev].groupby("system", sort=False):
            r1, ref = g[g.set == "rd1"], g[g.set == "ref"]
            fl = r1[r1.flagged]
            p = pf[(pf.system == s_) & (pf.revision == rev)]
            out[s_] = {"acc alone rd %": 100 * (r1.pred == r1.y).mean(),
                       "acc policy rd %": 100 * (r1.final == r1.y).mean(),
                       "acc policy clean %": 100 * (ref.final == ref.y).mean(),
                       "flagged rd %": 100 * r1.flagged.mean(), "false alarm clean %": 100 * ref.flagged.mean(),
                       "names right group, flagged %": 100 * (fl.named == fl.group).mean() if len(fl) else np.nan,
                       "AUROC rd vs clean": p.auroc.mean(), "folds in AUROC": int(p.auroc.notna().sum())}
        t = pd.DataFrame(out).T
        t.loc[SYS_CONF, "names right group, flagged %"] = np.nan
        print(t.round(3).to_string())

    print("\nREAL vs SIMULATED, REV3 (main), revision 1, by group: real dropout (rd) | r_fill on the clean clips (rf)")
    for gname in SENS:
        r1, rf = m[(m.set == "rd1") & (m.group == gname)], m[(m.set == "rf") & (m.group == gname)]
        if not len(r1):
            continue
        nm = lambda d: pct((d[d.flagged].named == d[d.flagged].group).mean()) if d.flagged.any() else "  n/a"
        print(f"  {gname:5s} (n rd {r1.clip.nunique()}): flagged rd {pct(r1.flagged.mean())} rf {pct(rf.flagged.mean())}"
              f" | names {gname} (flagged) rd {nm(r1)} rf {nm(rf)} | policy acc rd {pct((r1.final == r1.y).mean())}"
              f" rf {pct((rf.final == rf.y).mean())} | alone rd {pct((r1.pred == r1.y).mean())} rf {pct((rf.pred == rf.y).mean())}")


if __name__ == "__main__":
    main()
