# Data audit: MMAct V1 trimmed video (checkpoint 1, 5 Nov 2026)

Purpose: add video as a third MMAct sensor (watch + phone + video) for the journal version.
All decisions below were fixed on 5 Nov 2026, **before any video result existed**.

## 1. Source and extraction
- Raw zip: `MMActV1_trimmed_video.zip` (88,997,552,697 bytes; SHA-256 in `data/raw/MMAct/SHA256SUMS.txt`).
  The zips now live on the external backup drive (`Research Backups/TrustMI-HAR/data/raw/MMAct/`);
  the local `data/raw/MMAct/` folder is empty.
- The zip holds 20 archives, `trimmed/subject01.tar.gz` ... `subject20.tar.gz` (88,983,971,069 bytes unpacked).
- Extracted on 5 Nov 2026 by streaming each archive straight out of the zip into `tar`
  (`unzip -p ... | tar -xzf - -C data/extracted/MMAct`), so the `.tar.gz` files were never written to disk.
  Every subject finished with exit codes `0 0` (log: `data/extracted/MMAct/extract_log.txt`; subject 1 run by hand).
- Completeness check: per subject, the extracted bytes are 1.0005-1.0006 x the `.tar.gz` size (videos are
  already compressed, so gzip saves almost nothing); no 0-byte files. Total 35,162 videos, 89.03 GB.

## 2. Layout
- `subjectN/camK/sceneS/sessionM/<action>.mp4`: 20 subjects, 4 cameras, 4 scenes, 5 sessions.
  Folder names have no leading zero (`subject1`), unlike the archive names (`subject01`).
- Sensor clips are keyed by subject + scene + session + action, so **each sensor clip has up to 4 videos**,
  one per camera.
- Same 35 action names as the sensor data after lowercasing. Only `Carrying_light` (4 videos) is not lowercase.
  The class renames in the 2020 sensor update log do not appear in the video names either.

## 3. Video format (sample: subject 1, scene 1, session 1, carrying, all 4 cameras)
- H.264, 1920x1080, 29.97 fps.
- The 4 views of a clip have identical length (268 frames, 8.94 s): the cameras are recorded in sync.
- Video and sensor cover the same span (8.94 s video vs 8.95 s sensor), unlike UTD-MHAD where the IMU runs longer.
- Main sensor clips last 1.0-42.1 s (median 6.1 s, 95% under 15.1 s).
- Decoding cost: ~1.2-1.5 s of one CPU core per 9 s video, so ~10 core-hours for all videos;
  decoding, not the GPU, sets the speed of feature extraction.

## 4. Matching video to sensor clips
| Sensor clips (status from `mmact_sensor_v1.npz`) | 0 views | 1 | 2 | 3 | 4 views | Total |
|---|---|---|---|---|---|---|
| main | 25 | 25 | 50 | 511 | 7,860 | 8,471 |
| no_watch | 0 | 0 | 0 | 0 | 247 | 247 |
| excluded_length | 1 | 1 | 2 | 17 | 207 | 228 |
| excluded_rate | 0 | 0 | 0 | 0 | 35 | 35 |

- 92.8% of main clips have all 4 views. Per camera: cam1 8,241, cam2 8,395, cam3 8,270, cam4 8,192 of 8,471.
- The 25 main clips with no video are one whole block: subject 8, scene 2, session 5.
- All 247 no-watch clips have all 4 views, so with video they become two-sensor clips (phone + video).
- 13 video keys have no sensor clip (subjects 1, 2, 4, 5; e.g. subject 1/2 scene 4 session 5 fall,
  picking_up, pulling); they are not used.
- Missing views are spread across subjects; subject 8 has the most gaps (329-381 videos per camera).
- These numbers were first computed from the file listing and the sensor index; `scripts/audit_mmact_video.py`
  recomputes them from the files themselves (section 7).

## 5. Why the standard crop is not used
torchvision's preset for R(2+1)D-18 (`weights.transforms()`, used for the UTD-MHAD conference features) resizes
every frame to exactly 128x171 and then center-crops 112x112, keeping the central 65.5% of the width and
87.5% of the height.
- On UTD-MHAD (640x480, 4:3, subject standing centred in front of the Kinect) this is harmless.
- On MMAct (16:9, four corner cameras, people walking around the room) it squeezes the frame
  (people ~25% narrower) and can cut the person out: in subject 1 / carrying, the person is half outside
  the crop in cam2 and loses the head in cam4.

## 6. Decisions (fixed before any result)
1. **Unit = one sensor clip.** The video sensor of a clip is the **mean of the per-view features over the views
   that exist**. Per-view features are also stored, so a single-camera ablation is possible later.
2. **Clips with no video** (25 main clips) get status `no_video`: left out of 3-sensor training and reported,
   like `no_watch`. With decision 7, three-sensor experiments use 8,432 main clips (`main3`).
3. **32 frames** spread evenly over the whole clip (same as UTD-MHAD). If a video has fewer than 32 frames,
   evenly spaced rounding repeats frames instead of failing.
4. **One preprocessing rule for every dataset (option C):** whole frame, no crop, aspect ratio kept,
   resized to height 112 (MMAct 1920x1080 -> 112x199; UTD-MHAD 640x480 -> 112x149), then the same
   colour normalisation (mean/std) as `weights.transforms()`.
   - Consequence: **UTD-MHAD video features are re-extracted** (`video_r2plus1d18_v2`) and all UTD-MHAD
     experiments that use video are rerun. The conference features (`video_r2plus1d18_v1`, center crop) are kept
     unchanged for reproducibility of the conference paper.
   - The journal paper states that the video preprocessing changed from the conference version to keep the
     full field of view, so UTD-MHAD figures differ slightly.
5. **Backbone unchanged:** frozen torchvision R(2+1)D-18, Kinetics-400 weights, no fine-tuning; output =
   4 time-step tokens (512 each) + one pooled 512-vector per view.
6. **Video lies:** the donor video comes from the same subject and the same scene group ({1,2,4} or {3}),
   the whole multi-view feature is swapped, as for the sensor lies.
7. **Video/sensor length check (added after the script run, before any video result):** a main clip enters
   the 3-sensor set only if video length / sensor length lies in 0.9-1.1, the same 10% rule that
   `build_mmact_sensor.py` uses for `excluded_length`. 14 main clips fail (section 7); in at least 12 of them the
   video and the sensor clearly do not cover the same stretch of time (dataset trimming errors), so a planted lie
   could not be told apart from the clip's own mismatch. Encoded as column `status3` in
   `data/processed/mmact_video_clips.csv`; clips already excluded on the sensor side keep `excluded_sensor`.
8. **Like-for-like comparisons (revised 5 Nov, before any video result):** every MMAct model of the journal
   version, with 1, 2 or 3 sensors, is trained AND scored on the same `main3` clips (8,432) with the same
   per-fold normalisation (dataset `d2v`). This includes the single-sensor watch and phone witnesses and the
   watch + phone model (which is also the backup when video is dropped), so every comparison, witness and
   backup uses identical training and test clips. The conference runs (`d2`, 8,471 clips) are kept unchanged
   for reproducing the conference paper; they are not mixed with `d2v` results. The first version of this
   decision (keep the 2-sensor models, filter at scoring time) was dropped because those models were
   normalised with `d2` statistics, so `d2v` inputs would not have matched them exactly.
9. **Same time window as the sensors:** the MMAct sensor loader (`src/trustmi/data/mmact.py`) centre-crops clips
   longer than 15 s to their middle 15 s. A video longer than 15 s therefore also uses only its centred 15 s
   (the 32 frames are spread over that window), so all three MMAct sensors see the same moment. UTD-MHAD clips
   are all short and use the whole clip. The window used is stored per view (`view_win_start`, `view_win_end`).
10. **Dataset names in the code** (`scripts/train.py`): `d1` / `d2` = conference versions (unchanged);
   `d1v2` = UTD-MHAD with v2 (whole-frame) video features; `d2v` = MMAct `main3` with video as a 3rd sensor.
   New names mean new run IDs (e.g. `BASE_d2v_b4s3_...`), so no conference result is skipped or overwritten.

Implementation: `scripts/extract_video_features_v2.py` -> `data/features/video_r2plus1d18_v2.npz` (UTD-MHAD,
same arrays as v1) and `data/features/mmact_video_r2plus1d18_v2.npz` (MMAct: per-clip `tokens`/`pooled` = mean
over views, plus all `view_*` arrays; clips keyed by the same `trial` names as `mmact_sensor_v1.npz`).

## 7. Script check (`scripts/audit_mmact_video.py`)
Reads every video header (resolution, fps, frame count) and its first frame, matches videos to the sensor clips
in `mmact_sensor_v1.npz`, and writes `data/processed/mmact_video_index.csv` (one row per video) and
`data/processed/mmact_video_clips.csv` (one row per clip key).

First run, 5 Nov 2026 (before decision 7 was added):
- 35,162 videos; 0 fail to open; 0 unreadable first frames; 0 duplicate keys.
- All 1920x1080. Frame rate 29.97 (29,335), 29.96 (5,601), 29.95 (217), 29.99 (9): rounding around 29.97.
- Action names identical to the sensor data after lowercasing (`Carrying_light`: 4 videos).
- Video length 0.73-51.54 s (median 6.11 s, 95% under 15.08 s).
- Views-per-clip table identical to section 4: 8,446 main clips with at least one view; the 25 without video
  are all subject 8 / scene 2 / session 5; 13 video-only keys.
- Camera views of one clip agree in length: largest difference 0.032 s (about one frame); none above 0.5 s.
- Video vs sensor length (8,446 main clips): median ratio 1.007; video is typically 0.044 s longer than the
  sensor clip. 14 clips fall outside 0.9-1.1:

  | Subject | Scene | Session | Action | Sensor (s) | Video (s) | Video/sensor |
  |---|---|---|---|---|---|---|
  | 1 | 2 | 1 | standing | 3.96 | 10.28 | 2.60 |
  | 1 | 4 | 1 | opening | 8.70 | 12.21 | 1.40 |
  | 1 | 4 | 2 | fall | 4.02 | 5.31 | 1.32 |
  | 1 | 4 | 4 | talking_on_phone | 8.46 | 6.74 | 0.80 |
  | 1 | 4 | 5 | pushing | 6.38 | 11.72 | 1.84 |
  | 3 | 1 | 3 | carrying | 9.24 | 7.97 | 0.86 |
  | 4 | 4 | 3 | running | 4.60 | 7.07 | 1.54 |
  | 8 | 1 | 1 | jumping | 8.12 | 3.77 | 0.46 |
  | 8 | 1 | 1 | setting_down | 11.88 | 7.04 | 0.59 |
  | 8 | 1 | 5 | picking_up | 1.10 | 4.97 | 4.52 |
  | 10 | 2 | 1 | pushing | 8.34 | 9.18 | just over 1.10 |
  | 11 | 4 | 3 | carrying_heavy | 19.24 | 4.24 | 0.22 |
  | 12 | 1 | 5 | standing | 3.08 | 1.57 | 0.51 |
  | 14 | 2 | 5 | jumping | 2.82 | 3.10 | just over 1.10 |

- 8 videos have fewer than 32 frames = 2 clips: subject 18 / scene 2 / session 5 / pointing (22 frames,
  already excluded on the sensor side) and subject 8 / scene 4 / session 3 / jumping (30 frames, main clip;
  decision 3 repeats 2 frames).

Second run (adds `status3`, decision 7): expected `main3` 8,432 | `excluded_sensor` 263 | `no_watch3` 247 |
`no_video` 25 | `excluded_video_length` 14 | `no_sensor` 13 (total 8,994 clip keys).
