# Datasets

Raw downloads live in data/raw/ (git-ignored, read-only). Nothing in data/raw/ is ever modified;
all processing reads from it and writes to data/extracted/ or data/processed/.

## UTD-MHAD (D1)
- Source: authors' page at personal.utdallas.edu/~kehtar/UTD-MAD/ (public download, no request).
- Cite: Chen, Jafari, Kehtarnavaz, "UTD-MHAD: A Multimodal Dataset for Human Action Recognition
  Utilizing a Depth Camera and a Wearable Inertial Sensor", ICIP 2015.
- Retrieved: 2026-09-17. All zips pass unzip -t.
- Known issues found in our audit:
  - 861 trials (8 subjects x 27 actions x 4, minus 3 removed by the authors as corrupted).
  - RGB is 15 fps (documentation says 30); Kinect skeleton/depth run at ~19.5 fps.
  - Streams are clip-aligned, not frame-synchronous (IMU runs longer than RGB, median 0.12-0.18 s).
  - IMU is on the right wrist for actions 1-21 and the right thigh for actions 22-27.
- SHA-256 of raw zips:

    b1a22ebd299ab8f6322f98a0451bd356e377d5c2ab0b4f4df5ef967bb0549f0c  Depth.zip
    1900de05775c04626dee30b34638b79dd739d481786f0a788f2fbb00663abea4  Inertial.zip
    058dded078b2c21eff901d98fe0a86d5343d11e9d6c07081aaa3e3280a43052b  RGB.zip
    e52d17c2e4aeda21aa1b1d9d9c1a1ec95cc76763cbee1966da935dacf45c6301  Sample_Code.zip
    f029ea1d4e4c330b2c61307b6d532b34c3920bcac1f4c22c7827f406e2c7ac00  Skeleton.zip

## MMAct V1, trimmed (D2)
- Source: requested via the authors' Google Form; download links + password received by email.
- Cite: Kong et al., "MMAct: A Large-Scale Dataset for Cross Modal Human Action Understanding", ICCV 2019.
- Terms: academic, non-commercial use only; redistribution prohibited. The download links and
  password are personal to this request and are deliberately NOT recorded in this repository.
- Sections used: "MMActV1 Complete trimmed data" (trimmed sensor + trimmed video).
- Also downloaded but not used: MMActV2 trimmed + untrimmed video (crowd localisation, video only).
  MMAct Challenge 2021 split not downloaded.
- Retrieved: 2026-09-22 to 2026-09-23. All four zips pass unzip -t.
- Video layout: one tar.gz per subject (subject01 ... subject20), zero-padded, unlike sensor folders.
  Video archives are dated 2019-10-25; sensor data was updated 2020-01-17.
- Sensor update log (2020-01-17) mentions renaming two classes, but the names it gives
  (object_transferring, transferring_large_object, transfering_objects) do not appear in the sensor
  files, which use loitering and transferring_object. Video filenames still to be checked.
- SHA-256 of raw zips:

    45ee3afc6e17967c6c098bbd22a125c4988a1973e01f9c22cbe32dfd2683381e  MMActV1_trimmed_sensor.zip
    935d4bcdd5b3b131e0fdf5be4357e5728d84e86c6cda3aef8289e5ffeb39fdd5  MMActV1_trimmed_video.zip
    ee31f361ac3c562d817fc95a8b818177e0ac96a18a83cdb6341bc74ac494f917  MMActV2_trimmed_video.zip
    155f8cccc48ccb899f5869f4faf1e21f699226ea92a32e25adcfb8a1c8bed62e  MMActV2_untrimmed_video.zip

### MMAct sensor audit (scripts/audit_mmact_sensor*.py, scripts/check_mmact_sensor.py)
- Layout: <stream>/subjectN/sceneK/sessionM/<action>.csv; 20 subjects, 4 scenes, 5 sessions.
  Headerless CSV: wall-clock timestamp + 3 values. 35,685 files, 10 of them empty.
- Streams, rates, units (from device limits): acc_watch 100 Hz, m/s^2, saturates at +-8 g;
  acc_phone 100 Hz, m/s^2, saturates at ~+-4 g; gyro (phone) 50 Hz, rad/s, saturates at +-1000 deg/s;
  orientation (phone) 50 Hz, rad, probably heading/pitch/roll; heading and roll wrap at +-pi.
  Both accelerometers include gravity.
- Saturation is common on the phone accelerometer (~10% of clips), rarer on the watch (~3%), rare on gyro.
- Per-row timestamps are unreliable: whole-second rows lack the fractional part; timestamps repeat,
  step backwards, and jump by +-60 s / +-3600 s, while the sample values stay smooth in file order.
  File order is the true order (smoother than timestamp order in 97-100% of reordered files), and
  row counts match the nominal rates (median rows/span ratio 0.996-1.003).
- Scene 3 has its own action set (drinking, pocket_in, pocket_out, sitting, sitting_down, standing_up,
  using_pc); scenes 1, 2 and 4 share the same 26 other actions; talking_on_phone and using_phone occur
  in all four scenes.
- carrying_heavy and carrying_light exist only for subjects 11-20; no two clips in a session overlap
  in time, so they are separate recordings from "carrying". One clip is labelled "Carrying_light".
- Watch data is missing in whole-session blocks for 6 subjects (249 clips; subject 2 scene 4 almost entirely).

### MMAct sensor tensors v1 (scripts/build_mmact_sensor.py -> data/processed/mmact_sensor_v1.npz)
- Sample order = file order; per-row timestamps ignored.
- Common rate 50 Hz: accelerometers downsampled 100 -> 50 Hz (resample_poly, line-padded edges);
  gyro and orientation stored unchanged. All streams start at t = 0 (start offsets median 10 ms,
  max 75 ms) and are trimmed to the shortest stream.
- Lossless ragged storage (clip i = X[start[i] : start[i] + length[i]]); 12 channels, native units;
  windowing decided at load time (default 15 s, covers 95% of clips intact).
- Labels lowercased -> 35 classes.
- Clip status: main 8471 | no_watch 247 | excluded_length 228 (stream lengths differ by > 10%)
  | excluded_rate 35 (row count vs time span outside 0.8-1.25). List: data/processed/mmact_excluded_clips.csv.
- Rules for later stages: orientation converted to sin/cos at load time; semantic swaps only within
  the same scene group ({1,2,4} or {3}); swapped clips resampled to the target clip's length.

### MMAct splits (scripts/make_splits_mmact.py -> data/splits/mmact_loso_v1.json)
- 20-fold leave-one-subject-out: test = subject k, val = next subject, train = remaining 18.
- Main clips only; no_watch clips of the test subject form a separate test set.
- Folds testing subjects 1-10 contain 33 classes (no heavy/light carrying): macro-F1 is computed
  over the classes present in each test fold.
