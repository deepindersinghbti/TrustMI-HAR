"""Certificate v1b: ask the single-sensor EXPERT models about every clip and condition (no training).

Same folds, parts (val/test), seeds and conditions as cert_signals.py; the operators are deterministic,
so every input is identical (the scorer checks this by matching the fused answers row by row).
Per row, all label-free:
  exp_pred_<m>, exp_conf_<m>  the expert's own answer and confidence
  exp_support_<m>             how much the expert believes the FUSED model's answer (its prob for it)
  exp_js                      mean pairwise Jensen-Shannon divergence between the experts' distributions
  exp_disagree                True if the experts' answers are not all the same
  exp_div_<m>                 mean JS divergence of expert m from the OTHER experts ("odd one out")
Plus y, subject, attacked, donor_y, available, correct, msp, entropy, energy, so the file stands alone.
--extra-models b4_noskel,b4_novideo   also store model_pred_<name>, model_conf_<name> (e.g. option-A backups)
--attack imu,skel,video   plant lies in these sensors (default: the physical sensors, as in cert_signals.py);
                          harmless conditions only for sensors with a physical model. Adds _att-... to the name.
Output: results/certificate/experts_<dataset>_<folds>_s<seeds>[_att-...].parquet
"""
import argparse, itertools, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from eval_operators import load_model, dataset_cls, MODELS as FUSED
from cert_signals import CFG, run_dir, repro, fwd
from cert_conditions import conditions_for, conditions_suffix, parts_for, dataset_for, folds_for
from cert_evidential import load_ev, METHODS as EV_METHODS

EXPERTS = dict(d1=dict(imu="b1_imu", skel="b1_skel", video="b1_video"),
               d2=dict(watch="b1_watch", phone="b1_phone"),
               d1v2=dict(imu="b1_imu", skel="b1_skel", video="b1_video"),          # journal
               d2v=dict(watch="b1_watch", phone="b1_phone", video="b1_video"),
               d3=dict(arms="b1_arms", trunk="b1_trunk", legs="b1_legs"))                  # journal, WearGait-PD
EXTRA_MODS = dict(b4_noimu=["video", "skel"], b4_noskel=["video", "imu"], b4_novideo=["skel", "imu"],  # as in train.py
                  b4s_nowatch=["phone", "video"], b4s_nophone=["watch", "video"], b4s=["watch", "phone"],  # d2v backups
                  b4w_noarms=["trunk", "legs"], b4w_notrunk=["arms", "legs"], b4w_nolegs=["arms", "trunk"])  # d3


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(np.clip(a, 1e-12, 1)) - np.log(np.clip(b, 1e-12, 1)))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=CFG, required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--attack", default=None, help="comma list of sensors to swap")
    ap.add_argument("--extra-models", default="", help="comma list of extra models to store answers for")
    ap.add_argument("--conditions", choices=("lies", "faults", "nowatch", "realdrop", "imuoff", "nowatch2", "imuoff2", "faults2"), default="lies",
                    help="lies (default, unchanged), faults (docs/plan_faults_2026-11-11.md), "
                         "nowatch (d2v only, docs/plan_nowatch_2026-11-11.md), "
                         "realdrop (d3 only, docs/plan_weargait_2026-11-11.md) or "
                         "imuoff (d3 only, docs/plan_followup_2026-11-12.md)")
    ap.add_argument("--no-video-faults", action="store_true",
                    help="faults without the video faults (smoke run before the video fault features exist)")
    ap.add_argument("--main", default=None, help="judge this evidential run's answers instead of B4's (e.g. tmc, tmc_s)")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    C, dname = CFG[args.dataset], args.dataset
    if args.main and args.conditions in ("nowatch", "realdrop", "imuoff", "nowatch2", "imuoff2"):
        raise SystemExit(f"--main is not supported with --conditions {args.conditions}")
    mods, E = FUSED[C["model"]], EXPERTS[args.dataset]
    folds = list(range(C["n_folds"])) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    folds = folds_for(args.conditions, dname, folds)          # nowatch: only folds with no-watch clips
    seeds = [int(s) for s in args.seeds.split(",")]
    DS = dataset_cls(dname)
    out_dir = ROOT / "results/certificate"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.folds == "all" else "f" + "-".join(map(str, folds))
    attack = args.attack.split(",") if args.attack else C.get("attack", C["sensors"])
    suffix = (f"_att-{'-'.join(attack)}" if args.attack else "") + (f"_main-{args.main}" if args.main else "")
    suffix += conditions_suffix(args.conditions, not args.no_video_faults)
    out = out_dir / f"experts_{dname}_{tag}_s{'-'.join(map(str, seeds))}{suffix}.parquet"
    rows, t0 = [], time.time()

    for fold in folds:
        tr = DS(fold, "train")
        main = args.main or C["model"]
        if args.main:                                   # an evidential main model (TMC / ECML)
            meth, mmods = EV_METHODS[dname][args.main]
            fused = {s: load_ev(run_dir(dname, main, fold, s), meth, mmods, C["n_classes"]) for s in seeds}
        else:
            fused = {s: load_model(run_dir(dname, C["model"], fold, s), mods, C["n_classes"]) for s in seeds}
        experts = {(m, s): load_model(run_dir(dname, b, fold, s), [m], C["n_classes"])
                   for m, b in E.items() for s in seeds}
        extras = {(x, s): load_model(run_dir(dname, x, fold, s), EXTRA_MODS[x], C["n_classes"])
                  for x in filter(None, args.extra_models.split(",")) for s in seeds}
        for part in parts_for(args.conditions):
            ds = dataset_for(args.conditions, DS, dname, fold, part)
            base = {m: ds.x[m] for m in mods}
            lens = ds.x["len"] if "len" in ds.x else None
            y = np.asarray(ds.y).astype(int); N = len(y); ar = np.arange(N)
            conds, gidx = conditions_for(args.conditions, dname, ds, tr, fold, attack, C["sensors"],
                                         not args.no_video_faults)               # shared with cert_mi.py
            subj = np.asarray(ds.subject)
            for cname, (repl, donor_y, avail, attacked) in conds.items():
                inp = dict(base); inp.update({k: torch.from_numpy(v) for k, v in repl.items()})
                for s in seeds:
                    z0, p0 = fwd(fused[s], inp, lens)
                    if cname == "clean":
                        ok, msg = repro(run_dir(dname, main, fold, s), part, z0, y)
                        print(f"fold {fold:2d} {part:4s} seed {s}: fused clean | {msg}")
                        if not ok:
                            raise SystemExit("clean logits do not match the stored run: stop and paste this")
                    pred = p0.argmax(1)
                    P = {m: fwd(experts[(m, s)], inp, lens)[1] for m in E}
                    lse = np.log(np.exp(z0 - z0.max(1, keepdims=True)).sum(1)) + z0.max(1)
                    cols = dict(fold=fold, part=part, seed=s, clip=gidx, condition=cname, pred_fused=pred,
                                y=y, subject=subj, attacked=attacked, donor_y=donor_y, available=avail,
                                correct=pred == y, msp=p0.max(1),
                                entropy=-(p0 * np.log(np.clip(p0, 1e-12, 1))).sum(1), energy=-lse)
                    for m, pm in P.items():
                        cols[f"exp_pred_{m}"] = pm.argmax(1)
                        cols[f"exp_conf_{m}"] = pm.max(1)
                        cols[f"exp_support_{m}"] = pm[ar, pred]
                    if args.conditions in ("nowatch", "nowatch2"):    # M = watch flagged absent; scene
                        pm = fwd(fused[s], inp, lens, {"watch": torch.ones(N, dtype=torch.bool)})[1]
                        cols["mask_watch_pred"], cols["mask_watch_msp"] = pm.argmax(1), pm.max(1)
                        cols["scene"] = np.asarray(ds.group)
                    if args.conditions in ("realdrop", "imuoff", "imuoff2"):   # M = missing group(s) flagged absent
                        hide = set(attacked.split(",")) - {""}
                        pm = fwd(fused[s], inp, lens, {m: torch.full((N,), m in hide) for m in hide})[1] if hide else p0
                        cols["mask_missing_pred"], cols["mask_missing_msp"] = pm.argmax(1), pm.max(1)
                    cols["exp_js"] = np.mean([js(P[a], P[b]) for a, b in itertools.combinations(P, 2)], axis=0)
                    for (x, sx), mdl in extras.items():
                        if sx == s:
                            px = fwd(mdl, inp, lens)[1]
                            cols[f"model_pred_{x}"] = px.argmax(1)
                            cols[f"model_conf_{x}"] = px.max(1)
                    for m in P:
                        cols[f"exp_div_{m}"] = np.mean([js(P[m], P[k]) for k in P if k != m], axis=0)
                    ep = np.stack([P[m].argmax(1) for m in P])
                    cols["exp_disagree"] = (ep != ep[0]).any(0)
                    rows.append(pd.DataFrame(cols))
        print(f"fold {fold:2d}: done ({time.time() - t0:.0f} s elapsed)")

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out, index=False)
    t = df[df.part == "test"]
    print(f"\nSANITY ({dname}, test part) -> {out.relative_to(ROOT)}")
    print(t.groupby("condition", sort=False)[["exp_disagree", "exp_js"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
