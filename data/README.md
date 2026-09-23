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
- Also downloaded but not used: MMActV2 trimmed + untrimmed video (crowd spatial-temporal
  localisation, video only). MMAct Challenge 2021 split not downloaded.
- Retrieved: 2026-09-22 to 2026-09-23. All four zips pass unzip -t.
- Sensor layout: <stream>/subjectN/sceneK/sessionM/<action>.csv, headerless CSV,
  column 1 = wall-clock timestamp (YYYYMMDD_HH:MM:SS.ffffff), then 3 value columns.
  Streams: acc_phone (8982 files), gyro (8982), orientation (8985), acc_watch (8736).
- Video layout: one tar.gz per subject (subject01 ... subject20), zero-padded, unlike sensor folders.
- Sensor update log (2020-01-17) renames two classes: object_transferring -> loitering,
  transferring_large_object -> transfering_objects. Video archives are dated 2019-10-25,
  i.e. before this fix, so video filenames may still use the old names.
- SHA-256 of raw zips:

    45ee3afc6e17967c6c098bbd22a125c4988a1973e01f9c22cbe32dfd2683381e  MMActV1_trimmed_sensor.zip
    935d4bcdd5b3b131e0fdf5be4357e5728d84e86c6cda3aef8289e5ffeb39fdd5  MMActV1_trimmed_video.zip
    ee31f361ac3c562d817fc95a8b818177e0ac96a18a83cdb6341bc74ac494f917  MMActV2_trimmed_video.zip
    155f8cccc48ccb899f5869f4faf1e21f699226ea92a32e25adcfb8a1c8bed62e  MMActV2_untrimmed_video.zip
