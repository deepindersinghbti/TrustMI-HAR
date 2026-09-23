"""MMAct sensor checks before building tensors: sample order, stream-length agreement,
overlapping clips, carrying classes by subject, clip durations."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path("data/extracted/MMAct/sensor/trimmed")
FS = {"acc_phone_clip": 100, "acc_watch_clip": 100, "gyro_clip": 50, "orientation_clip": 50}
KEY = ["subject", "scene", "session", "action"]


def parse_ts(raw):
    t = pd.to_datetime(raw, format="%Y%m%d_%H:%M:%S.%f", errors="coerce")
    miss = t.isna()
    t[miss] = pd.to_datetime(raw[miss], format="%Y%m%d_%H:%M:%S", errors="coerce")
    return t


def order_test(rel):
    """Is the signal smoother in file order or in timestamp order?"""
    d = pd.read_csv(ROOT / rel, header=None)
    t = parse_ts(d[0].astype(str)).to_numpy()
    v = d.iloc[:, 1:].to_numpy(float)
    order = np.argsort(t, kind="stable")
    rough = lambda x: np.abs(np.diff(x, axis=0)).mean()
    return dict(file=rel, moved=int((order != np.arange(len(order))).sum()),
                rough_file=rough(v), rough_time=rough(v[order]))


if __name__ == "__main__":
    idx = pd.read_parquet("data/processed/mmact_sensor_index.parquet")
    a2 = pd.read_parquet("data/processed/mmact_sensor_audit2.parquet")
    ok = idx[idx.error.isna()].copy()
    ok["action"] = ok.action.str.lower()

    with ProcessPoolExecutor() as ex:
        o = pd.DataFrame(ex.map(order_test, ok.file.tolist(), chunksize=64))
    o["stream"] = o.file.str.split("/").str[0]
    o["ratio"] = o.rough_time / o.rough_file
    print("1) SAMPLE ORDER: is the signal smoother in file order than in timestamp order?")
    for s, g in o.groupby("stream"):
        m = g[g.moved > 0]
        print(f"   {s:16s} files where sorting changes the order {len(m):5d}/{len(g)} | "
              f"file order smoother in {(m.ratio > 1).mean():.1%} of them | "
              f"median roughness ratio {m.ratio.median():.2f}")

    print("\n2) FILES WHOSE ROW COUNT DISAGREES WITH THEIR TIME SPAN")
    for s, g in a2.groupby("stream"):
        r = g.rate_ratio
        print(f"   {s:16s} outside 0.9-1.1: {int(((r < .9) | (r > 1.1)).sum()):4d} | "
              f"outside 0.8-1.25: {int(((r < .8) | (r > 1.25)).sum()):4d}")

    ok["dur_samples"] = ok.n_rows / ok.stream.map(FS)
    w = ok.pivot_table(index=KEY, columns="stream", values="dur_samples").dropna()
    spread = w.max(axis=1) / w.min(axis=1)
    print(f"\n3) STREAM-LENGTH AGREEMENT over {len(w)} complete clips (longest / shortest stream)")
    print("  ", spread.describe(percentiles=[.5, .9, .99]).round(3).to_dict())
    print(f"   clips with spread > 1.10: {int((spread > 1.10).sum())} | > 1.25: {int((spread > 1.25).sum())}")

    ph = ok[ok.stream == "acc_phone_clip"].dropna(subset=["t_start", "t_end"]).sort_values("t_start")
    pairs = []
    for _, g in ph.groupby(["subject", "scene", "session"]):
        s, e, a = g.t_start.to_numpy(), g.t_end.to_numpy(), g.action.to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if s[j] >= e[i]:
                    break
                inter = (min(e[i], e[j]) - s[j]) / np.timedelta64(1, "s")
                short = min(e[i] - s[i], e[j] - s[j]) / np.timedelta64(1, "s")
                pairs.append(dict(pair=" + ".join(sorted([a[i], a[j]])), frac=inter / short))
    p = pd.DataFrame(pairs)
    print(f"\n4) OVERLAPPING CLIPS WITHIN A SESSION (phone stream): {len(p)} pairs")
    if len(p):
        print(p.groupby("pair").frac.agg(["size", "median"]).sort_values("size", ascending=False)
              .head(10).round(2).to_string())

    c = ph[ph.action.str.startswith("carrying")]
    print("\n5) CARRYING CLIPS PER SUBJECT")
    print(pd.crosstab(c.action, c.subject).to_string())

    d = ph.n_rows / 100
    print("\n6) CLIP DURATION (s, from sample count):",
          d.describe(percentiles=[.1, .5, .9, .95, .99]).round(2).to_dict())
    for lim in (5, 8, 10, 15):
        print(f"   clips <= {lim:2d} s: {(d <= lim).mean():.1%}")
