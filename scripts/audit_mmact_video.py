"""Audit MMAct V1 trimmed video and match it to the sensor clips (checkpoint 1, 2026-11-05).

Reads only: data/extracted/MMAct/subject*/cam*/scene*/session*/<action>.mp4 (headers + first frame)
            data/processed/mmact_sensor_v1.npz (clip labels and lengths only, not the signals)
Writes:     data/processed/mmact_video_index.csv  one row per video file
            data/processed/mmact_video_clips.csv  one row per clip key (sensor clips + video-only keys)
Decisions are recorded in docs/data_audit_mmact_video.md.

Run from the project root:  python scripts/audit_mmact_video.py
"""
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

ROOT = Path("data/extracted/MMAct")
SENSOR = "data/processed/mmact_sensor_v1.npz"
OUT_VID = Path("data/processed/mmact_video_index.csv")
OUT_CLIP = Path("data/processed/mmact_video_clips.csv")
KEY = ["subject", "scene", "session", "action"]
PAT = re.compile(r"subject(\d+)/cam(\d)/scene(\d)/session(\d)/([^/]+)\.mp4$")
T = 32  # frames per clip planned for the video features
RATIO_LO, RATIO_HI = 0.9, 1.1  # video/sensor duration band for the 3-sensor set (decision 7)


def probe(path):
    """Header values + whether the first frame decodes. Does not decode the whole file."""
    cap = cv2.VideoCapture(str(path))
    r = dict(opened=bool(cap.isOpened()),
             width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             fps=float(cap.get(cv2.CAP_PROP_FPS)),
             n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    ok, frame = cap.read() if r["opened"] else (False, None)
    r["first_frame_ok"] = bool(ok and frame is not None and frame.size > 0)
    cap.release()
    return r


if __name__ == "__main__":
    # 1. every video file
    rows = []
    for p in sorted(ROOT.glob("subject*/cam*/scene*/session*/*.mp4")):
        sj, cam, sc, se, act = PAT.search(p.as_posix()).groups()
        rows.append(dict(subject=int(sj), cam=int(cam), scene=int(sc), session=int(se),
                         action_raw=act, action=act.lower(),
                         path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size))
    v = pd.DataFrame(rows)
    workers = min(16, os.cpu_count() or 1)
    with ProcessPoolExecutor(workers) as ex:
        pr = list(ex.map(probe, [ROOT / x for x in v.path], chunksize=64))
    v = pd.concat([v, pd.DataFrame(pr)], axis=1)
    v["duration_s"] = np.where(v.fps > 0, v.n_frames / v.fps.where(v.fps > 0, 1), np.nan)
    v.to_csv(OUT_VID, index=False)

    # 2. sensor clips (labels only)
    d = np.load(SENSOR)
    s = pd.DataFrame(dict(subject=d["subject"].astype(int), scene=d["scene"].astype(int),
                          session=d["session"].astype(int), action=d["action"].astype(str),
                          trial=d["trial"].astype(str), sensor_status=d["status"].astype(str),
                          sensor_s=d["length"] / float(d["fs"])))

    # 3. one row per clip key: which camera views exist and how long they are
    good = v[v.opened & v.first_frame_ok]
    g = good.groupby(KEY)
    vc = pd.DataFrame(dict(
        cams=g.cam.agg(lambda c: "".join(map(str, sorted(c)))),
        n_cams=g.cam.nunique(),
        video_s=g.duration_s.median(),
        cam_spread_s=g.duration_s.agg(lambda x: x.max() - x.min()),
        frames_min=g.n_frames.min())).reset_index()
    c = s.merge(vc, on=KEY, how="outer", indicator=True)
    c["n_cams"] = c.n_cams.fillna(0).astype(int)
    c["sensor_status"] = c.sensor_status.fillna("no_sensor")
    c["video_status"] = np.where(c.n_cams > 0, "ok", "no_video")
    c["video_over_sensor"] = c.video_s / c.sensor_s
    # 3-sensor clip set (decision 7): same 10% rule as excluded_length in build_mmact_sensor.py
    len_ok = c.video_over_sensor.between(RATIO_LO, RATIO_HI)
    c["status3"] = np.select(
        [c.sensor_status == "no_sensor", c.sensor_status.str.startswith("excluded"), c.n_cams == 0,
         ~len_ok, c.sensor_status == "main", c.sensor_status == "no_watch"],
        ["no_sensor", "excluded_sensor", "no_video", "excluded_video_length", "main3", "no_watch3"],
        default="other")
    c.drop(columns="_merge").sort_values(KEY).to_csv(OUT_CLIP, index=False)

    # 4. report
    print(f"video files: {len(v)} | could not open: {(~v.opened).sum()} | first frame unreadable: "
          f"{(v.opened & ~v.first_frame_ok).sum()} | duplicate keys: "
          f"{v.duplicated(['subject', 'cam'] + KEY[1:]).sum()}")
    print("resolution:", (v.width.astype(str) + "x" + v.height.astype(str)).value_counts().to_dict())
    print("fps:", v.fps.round(3).value_counts().to_dict())
    print("names not lowercase:", v.loc[v.action_raw != v.action, "action_raw"].value_counts().to_dict())
    print("actions only in video:", sorted(set(v.action) - set(s.action)),
          "| only in sensor:", sorted(set(s.action) - set(v.action)))
    print(f"videos with fewer than {T} frames: {(v.n_frames < T).sum()}")
    print("video duration (s):", v.duration_s.describe(percentiles=[.05, .5, .95]).round(2).to_dict())
    print("\nclips: sensor status x number of camera views")
    print(pd.crosstab(c.sensor_status, c.n_cams, margins=True))
    main = c[(c.sensor_status == "main") & (c.n_cams > 0)]
    print("\nmain clips with >= 1 view:", len(main),
          "| per camera:", {k: int(main.cams.str.contains(str(k)).sum()) for k in range(1, 5)})
    print("no-video main clips by subject/scene/session:",
          c[(c.sensor_status == "main") & (c.n_cams == 0)].groupby(["subject", "scene", "session"]).size().to_dict())
    print("\nlength difference between camera views of one clip (s):",
          main.cam_spread_s.describe(percentiles=[.5, .95, .99]).round(3).to_dict())
    print("clips whose views differ by > 0.5 s:", int((main.cam_spread_s > 0.5).sum()))
    print("video / sensor duration:", main.video_over_sensor.describe(percentiles=[.01, .05, .5, .95, .99]).round(3).to_dict())
    print("clips with video/sensor outside 0.9-1.1:", int(((main.video_over_sensor < 0.9) | (main.video_over_sensor > 1.1)).sum()))
    print("\n3-sensor status (status3):", c.status3.value_counts().to_dict())
    print("excluded_video_length clips:")
    print(c[c.status3 == "excluded_video_length"][KEY + ["sensor_status", "sensor_s", "video_s", "video_over_sensor"]]
          .round(2).to_string(index=False))
    print(f"\nsaved {OUT_VID} and {OUT_CLIP}")
