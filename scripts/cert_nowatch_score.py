"""No-watch plan, step 4 (docs/plan_nowatch_2026-11-11.md): score every system on the 247 MMAct clips where the
watch was never recorded.

  python scripts/cert_nowatch_score.py                    full run (queue G files, tag all_s0-1-2)
  python scripts/cert_nowatch_score.py --tag f19_s0       smoke-run files (code check only; never overwrites)

FROZEN, as in the fault test: every system is fitted on each fold's VALIDATION person with the LIE bench only
(cert_fault_score.Frozen, imported) and applied unchanged to the test person's no-watch bench:
  nw   nw_fill:watch   the no-watch clips, watch = its training mean (the realistic SILENT failure, condition F)
  ref  clean           the test person's reference clips (plan section 6): normal clips of the SAME scene and
                       actions as their no-watch clips; a person with no normal clip in that scene (person 2)
                       uses all their normal clips of the same actions and is labelled "no same-scene reference"
  rf   r_fill:watch    the same reference clips with the watch set to its training mean (question 2)
Answers without the certificate (condition K = known missing, M = flagged absent; plan section 3):
  K   backup b4s_nowatch (phone + video), and the dropout model with the watch hidden
  M   the standard model with the watch flagged absent, and the dropout model with the watch hidden (= its K)
Main result, named in advance (plan section 4): REV3's use / drop accuracy on nw, revision 1, against the full
model doing nothing on nw and against K.
Reporting (plan section 7 and implementation notes): per person; pooled over all clips for accuracy, flag and
naming; AUROC and "answer the safest 80%" only per person and as the plain mean over the four people with
>= 28 clips (scores from different folds are not on one scale); persons with < 28 clips as counts.
CHECKS before any number: (1) the bench's clean rows equal the lie bench's clean rows; (2) each frozen fit
reproduces the stored lie numbers exactly (cert_fault_score.check_frozen_on_lies).
Writes results/certificate/nowatch_d2v[_<tag>]_per_subject.csv and nowatch_d2v[_<tag>]_rows.parquet
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
from cert_score import auroc
from cert_fault_score import (Frozen, check_clean, check_frozen_on_lies, with_dropout, merge, FULL, KEYS,
                              SYS_CONF, SYS_C3, SYS_MAIN, SYS_D, REV3_SIG, REV3_RULES)

DS_ = "d2v"
NW, RF, CLEAN = "nw_fill:watch", "r_fill:watch", "clean"
MIN_BIG = 28                                  # plan section 7: persons with >= 28 no-watch clips
SETS = ("nw", "ref", "rf")


# ---------------------------------------------------------------- data
def bench(tag):
    ex = pd.read_parquet(RES / f"experts_{DS_}_{tag}_nowatch.parquet")
    ex = ex.assign(pred=ex.pred_fused)
    mi = pd.read_parquet(RES / f"mi_{DS_}_own_{tag}_nowatch.parquet")
    return merge(ex, mi, [c for c in mi.columns if c.startswith(("pmi_", "mi_", "s4"))], f"{DS_} no-watch MI")


def reference_sets(b):
    """Per row: 'nw', 'ref', 'rf' or '' (a normal clip outside the reference). Also the per-person note."""
    cond, subj = b.condition.to_numpy(), b.subject.to_numpy()
    scene, y = b.scene.to_numpy(), b.y.to_numpy()
    out = np.full(len(b), "", dtype=object)
    out[cond == NW] = "nw"
    notes = {}
    for p in np.unique(subj[cond == NW]):
        k_nw = (subj == p) & (cond == NW)
        scenes, actions = set(scene[k_nw]), set(y[k_nw])
        normal = (subj == p) & np.isin(cond, [CLEAN, RF]) & np.isin(y, list(actions))
        same = normal & np.isin(scene, list(scenes))
        sel = same if same.any() else normal
        notes[p] = "same scene and actions" if same.any() else "no same-scene reference (other scenes, same actions)"
        out[sel & (cond == CLEAN)] = "ref"
        out[sel & (cond == RF)] = "rf"
    return out, notes


def add_scene(df, b):
    """TMC / ECML files carry no scene column: take it from the experts bench, clip by clip."""
    sc = b[KEYS + ["scene"]]
    n = len(df)
    df = df.merge(sc, on=KEYS, how="inner", validate="many_to_one")
    if len(df) != n:
        raise SystemExit(f"{n - len(df)} evidential rows have no experts row: stop and paste this")
    return df


# ---------------------------------------------------------------- per-row results
def row_frame(te, D, named, final, A, t, system, rev, sets):
    k = sets != ""
    return pd.DataFrame(dict(
        system=system, revision=rev, fold=te.fold.to_numpy()[k], subject=te.subject.to_numpy()[k],
        seed=te.seed.to_numpy()[k], clip=te["clip"].to_numpy()[k], set=sets[k], y=te.y.to_numpy()[k],
        pred=te.pred.to_numpy()[k], final=final[k], score=D[k], abstain_score=A[k], flagged=D[k] >= t,
        named=named[k], msp=te.msp.to_numpy()[k]))


def answer_frame(b, sets):
    """Answers that need no certificate (plan section 3), standard and dropout model."""
    k = sets != ""
    cols = dict(fold=b.fold, subject=b.subject, seed=b.seed, clip=b["clip"], y=b.y,
                full_alone=b.pred, known_backup=b.model_pred_b4s_nowatch, flagged_absent=b.mask_watch_pred,
                dropout_alone=b.sd50_pred, dropout_hidden=b.sd50_hide_pred_watch)
    return pd.DataFrame({c: np.asarray(v)[k] for c, v in cols.items()}).assign(set=sets[k])


# ---------------------------------------------------------------- metrics
def person_metrics(r):
    """One dict per (system, revision, person) from the per-row frame r of that group."""
    s, y, f, p = r.set.to_numpy(), r.y.to_numpy(), r.final.to_numpy(), r.pred.to_numpy()
    D, A, fl, nm, msp = (r.score.to_numpy(), r.abstain_score.to_numpy(), r.flagged.to_numpy(),
                         r.named.to_numpy(), r.msp.to_numpy())
    out = {}
    for name, m in (("nw", s == "nw"), ("ref", s == "ref"), ("rf", s == "rf")):
        out[f"n_{name}"] = int(m.sum())
        out[f"correct_alone_{name}"] = int((p[m] == y[m]).sum())
        out[f"correct_policy_{name}"] = int((f[m] == y[m]).sum())
        out[f"n_flagged_{name}"] = int(fl[m].sum())
        out[f"n_named_watch_{name}"] = int((nm[m] == "watch").sum())
        out[f"n_named_watch_flagged_{name}"] = int(((nm[m] == "watch") & fl[m]).sum())
    for name, pos in (("nw", s == "nw"), ("rf", s == "rf")):
        both = pos | (s == "ref")
        out[f"auroc_{name}"] = auroc(D[both], pos[both])
        out[f"cov80_policy_{name}"] = CP.selective(A[both], f[both], y[both], 0.8)
        out[f"cov80_conf_{name}"] = CP.selective(1 - msp[both], p[both], y[both], 0.8)
    return out


def rate(df, num, den):
    n, d = df[num].sum(), df[den].sum()
    return np.nan if d == 0 else n / d


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=FULL, help=f"{FULL} = the full run; e.g. f19_s0 = smoke-run files (code check)")
    args = ap.parse_args(); ds = DS_
    MS.journal_files(ds)                                   # lie files: always the full run
    pd.set_option("display.width", 220)

    # ---- lie bench (fitting side) and no-watch bench (applying side)
    b = bench(args.tag)
    folds, seeds = sorted(b.fold.unique()), sorted(b.seed.unique())
    keep = lambda d: d[d.fold.isin(folds) & d.seed.isin(seeds)]
    lie = keep(MS.add_mi(CP.load(ds), ds))
    lie = lie[lie.available]
    check_clean(lie, b[b.part == "test"], ["pred", "msp", "exp_js", "s4"] + [f"exp_support_{m}" for m in CP.EXP[ds]],
                f"{ds} B4")
    if not (b.loc[b.condition == NW, "part"] == "nowatch").all() or not b.available.all():
        raise SystemExit("no-watch bench: unexpected part or unavailable rows: stop and paste this")
    ev_l = keep(pd.read_parquet(RES / f"{EVFILE[ds]}.parquet"))
    ev_l = ev_l[ev_l.available]
    ev_b = add_scene(pd.read_parquet(RES / f"evidential_{ds}_{args.tag}_nowatch.parquet"), b)
    for m in MS.MAIN_EV[ds]:
        check_clean(ev_l[ev_l.method == m], ev_b[(ev_b.method == m) & (ev_b.part == "test")],
                    ["pred", "msp", "u", "conflict"], f"{ds} {m}")
    d_l = with_dropout(lie, RES / f"dropout_{ds}_sd50_{FULL}.parquet", ds)
    b_raw = b.copy()
    d_b = with_dropout(b, RES / f"dropout_{ds}_sd50_{args.tag}_nowatch.parquet", ds)
    check_clean(d_l, d_b[d_b.part == "test"], ["pred", "msp"], f"{ds} dropout sd50")
    b_raw = b_raw.merge(d_b[KEYS + ["sd50_pred", "sd50_hide_pred_watch"]], on=KEYS, validate="one_to_one")

    sets_b, notes = reference_sets(b_raw)
    rev3 = MS.signals(REV3_SIG)
    systems = {   # name -> (lie df, bench df, features, naming rules, drop)
        SYS_CONF: (lie, b, MS.signals(["s1"]), None, None),
        SYS_C3: (lie, b, MS.signals(["s1", "s2", "s3"]), {"odd one out": MS.ODD}, CP.backup_answer),
        SYS_MAIN: (lie, b, rev3, REV3_RULES, CP.backup_answer),
        **{f"{m.upper()} alone": (ev_l[ev_l.method == m], ev_b[ev_b.method == m], ev_feats, EV_RULES, ev_drop)
           for m in MS.MAIN_EV[ds]},
        SYS_D: (d_l, d_b, rev3, REV3_RULES, DSC.hide_answer("sd50_hide_pred")),
    }

    rows = []
    for name_, (L, B, feats, rules, drop) in systems.items():
        sets_sys, _ = reference_sets(B)
        for rev in (1, 2):
            ref = per_subject(L, ds, feats, rules or {}, drop or (lambda d, ds_, n: d.pred.to_numpy()), rev) \
                if rules else None
            for fold in folds:
                va, te_l = L[(L.fold == fold) & (L.part == "val")], L[(L.fold == fold) & (L.part == "test")]
                kb = (B.fold == fold).to_numpy()
                te_b = B[kb]
                fz = Frozen(va, ds, feats, rules, drop, rev)
                if ref is not None:
                    check_frozen_on_lies(fz, te_l, ref.loc[te_l.subject.iloc[0]], f"{ds} {name_} rev {rev} fold {fold}")
                rows.append(row_frame(te_b, *fz.apply(te_b), fz.t, name_, rev, sets_sys[kb]))
        print(f"scored: {name_}" + (" (frozen fit reproduces the lie numbers exactly)" if rules else ""))
    R = pd.concat(rows, ignore_index=True)
    ans = answer_frame(b_raw, sets_b)

    # ---- the same rows in every system (same clips, seeds and reference selection)
    key = ["fold", "seed", "clip", "set"]
    base = R[(R.system == SYS_MAIN) & (R.revision == 1)][key].sort_values(key).reset_index(drop=True)
    for (s_, r_), g in R.groupby(["system", "revision"]):
        if not g[key].sort_values(key).reset_index(drop=True).equals(base):
            raise SystemExit(f"{s_} rev {r_}: different rows from REV3: stop and paste this")
    print(f"check: every system scored on the same {len(base)} rows "
          f"({(base.set == 'nw').sum()} no-watch, {(base.set == 'ref').sum()} reference, {(base.set == 'rf').sum()} r_fill)")

    per = pd.DataFrame([dict(system=s_, revision=r_, subject=p, fold=int(g.fold.iloc[0]), **person_metrics(g))
                        for (s_, r_, p), g in R.groupby(["system", "revision", "subject"], sort=False)])
    per["reference"] = per.subject.map(notes)
    report(per, ans, notes)
    sfx = "" if args.tag == FULL else f"_{args.tag}"
    out = RES / f"nowatch_{ds}{sfx}_per_subject.csv"
    per.to_csv(out, index=False)
    R.to_parquet(RES / f"nowatch_{ds}{sfx}_rows.parquet", index=False)
    print(f"\nsaved: {out.relative_to(ROOT)} and nowatch_{ds}{sfx}_rows.parquet")


# ---------------------------------------------------------------- report
def report(per, ans, notes):
    n_clips = per[per.system == SYS_MAIN].groupby("subject").n_nw.first()
    seeds = ans.seed.nunique()
    big = sorted(int(p) for p in notes if n_clips[p] // seeds >= MIN_BIG)
    small = sorted(int(p) for p in notes if p not in big)
    mb = f"mean of {len(big)}"
    print("\nPEOPLE: " + "; ".join(f"person {p}: {n_clips[p] // seeds} clips, reference = {notes[p]}" for p in sorted(notes)))
    print(f"  large people (>= {MIN_BIG} clips): {big}; counts only: {small}; {seeds} seeds")

    # ---- answers without a certificate
    print("\nANSWERS WITHOUT A CERTIFICATE, accuracy % on the no-watch clips (and on the reference clips)")
    cols = {"full model, watch filled (alone)": "full_alone", "K backup b4s_nowatch (phone + video)": "known_backup",
            "M full model, watch flagged absent": "flagged_absent", "D dropout model, watch filled": "dropout_alone",
            "D dropout model, watch hidden (= its K and M)": "dropout_hidden"}
    tab = {}
    for label, c in cols.items():
        row = {}
        for s in ("nw", "ref"):
            a = ans[ans.set == s]
            ok = (a[c] == a.y)
            g = ok.groupby(a.subject)
            for p in big:
                row[(s, f"p{p}")] = 100 * g.mean().get(p, np.nan)
            for p in small:
                row[(s, f"p{p}")] = f"{int(g.sum().get(p, 0))}/{int(g.size().get(p, 0))}"
            row[(s, "pooled")] = 100 * ok.mean()
            row[(s, mb)] = 100 * g.mean().reindex(big).mean()
        tab[label] = row
    t = pd.DataFrame(tab).T
    print(t.map(lambda v: v if isinstance(v, str) else f"{v:.1f}").to_string())
    print("  (reference column for person 2: other scenes, not comparable as a 'harder sessions' check)")

    # ---- main result
    print("\nMAIN RESULT (named in advance): REV3 (main), revision 1, accuracy % on the no-watch clips")
    m = per[(per.system == SYS_MAIN) & (per.revision == 1)].set_index("subject")
    k = ans[ans.set == "nw"]
    kg = (k.known_backup == k.y).groupby(k.subject)
    for p in sorted(notes):
        a, n = m.loc[p], m.loc[p, "n_nw"]
        f = (lambda c: f"{100 * c / n:5.1f}") if p in big else (lambda c: f"{int(c)}/{n}")
        print(f"  person {p:2d}: doing nothing {f(a.correct_alone_nw)} | REV3 policy {f(a.correct_policy_nw)}"
              f" | known missing (backup) {f(kg.sum()[p])}")
    tot = lambda c: 100 * m[c].sum() / m.n_nw.sum()
    mean4 = lambda c: 100 * (m.loc[big, c] / m.loc[big, "n_nw"]).mean()
    print(f"  pooled   : doing nothing {tot('correct_alone_nw'):5.1f} | REV3 policy {tot('correct_policy_nw'):5.1f}"
          f" | known missing {100 * kg.sum().sum() / kg.size().sum():5.1f}")
    print(f"  {mb}: doing nothing {mean4('correct_alone_nw'):5.1f} | REV3 policy {mean4('correct_policy_nw'):5.1f}"
          f" | known missing {100 * (kg.mean().reindex(big)).mean():5.1f}")

    # ---- every system
    for rev in (1, 2):
        print(f"\nSYSTEMS, revision {rev}" + (" (post hoc, labelled)" if rev == 2 else "")
              + f": no-watch clips (nw) vs reference clips (ref); pooled = all clips; AUROC and 80% = {mb} (large people)")
        out = {}
        for s_, g in per[per.revision == rev].groupby("system", sort=False):
            gb = g.set_index("subject").loc[big]
            out[s_] = {
                "acc alone nw %": 100 * rate(g, "correct_alone_nw", "n_nw"),
                "acc policy nw %": 100 * rate(g, "correct_policy_nw", "n_nw"),
                "acc policy ref %": 100 * rate(g, "correct_policy_ref", "n_ref"),
                "flagged nw %": 100 * rate(g, "n_flagged_nw", "n_nw"),
                "false alarm ref %": 100 * rate(g, "n_flagged_ref", "n_ref"),
                "names watch, flagged nw %": 100 * rate(g, "n_named_watch_flagged_nw", "n_flagged_nw"),
                "names watch, all nw %": 100 * rate(g, "n_named_watch_nw", "n_nw"),
                "AUROC nw vs ref": gb.auroc_nw.mean(),
                "safest 80% policy": 100 * gb.cov80_policy_nw.mean(),
                "safest 80% confidence": 100 * gb.cov80_conf_nw.mean(),
            }
        t = pd.DataFrame(out).T
        t.loc[SYS_CONF, ["names watch, flagged nw %", "names watch, all nw %"]] = np.nan   # it never names
        print(t.round(3).to_string())

    # ---- question 2: held-out sessions vs r_fill on the reference clips
    print("\nQUESTION 2, REV3 (main), revision 1: real no-watch clips (nw) next to r_fill of the watch on the"
          " reference clips (rf)")
    for p in sorted(notes):
        a = m.loc[p]
        if p in big:
            pct = lambda c, n: f"{100 * a[c] / a[n]:5.1f}" if a[n] else "  n/a"
        else:                                          # small person: counts, no AUROC (plan section 7)
            pct = lambda c, n: f"{int(a[c])}/{int(a[n])}"
            a = a.copy(); a["auroc_nw"] = a["auroc_rf"] = np.nan
        lab = "" if notes[p].startswith("same") else "  [no same-scene reference]"
        print(f"  person {p:2d}: flagged nw {pct('n_flagged_nw', 'n_nw')} rf {pct('n_flagged_rf', 'n_rf')} | names watch"
              f" (flagged) nw {pct('n_named_watch_flagged_nw', 'n_flagged_nw')} rf {pct('n_named_watch_flagged_rf', 'n_flagged_rf')}"
              f" | policy acc nw {pct('correct_policy_nw', 'n_nw')} rf {pct('correct_policy_rf', 'n_rf')}"
              f" | AUROC nw {a.auroc_nw:.3f} rf {a.auroc_rf:.3f}{lab}")


if __name__ == "__main__":
    main()
