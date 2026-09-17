# Data audit: UTD-MHAD (v1)

## Source and integrity
- Downloaded 2026-09-17 from personal.utdallas.edu/~kehtar/UTD-MAD/ (RGB, Depth, Skeleton, Inertial, Sample_Code zips).
- All zips pass `unzip -t`; SHA-256 in data/raw/UTD-MHAD/SHA256SUMS.txt; raw folder read-only.
- 861 trials = 27 actions x 8 subjects x 4 reps minus 3 removed by authors:
  a8_s1_t4, a23_s6_t4, a27_s8_t4. Classes have 31-32 trials each.
- Skeleton folder has 862 files: the extra is a copy of Skeleton_joint_order.txt (not data).
- No missing modality, no unreadable video, no NaN in IMU, no all-zero skeleton frames.

## Modalities as found on disk
- IMU `d_iner` (n, 6): cols 1-3 acc [g], cols 4-6 gyro [deg/s], 50 Hz, 107-326 samples.
- Skeleton `d_skel` (20, 3, n): Kinect v1 joint order, 41-125 frames.
- Depth `d_depth` (240, 320, n): same frame count as skeleton.
- RGB .avi: 640x480, 15 fps (not 30 as the website states), 32-96 frames.
- No timestamps in any file.

## Timing
- Skeleton rate implied by RGB duration: 19.5 +/- 1.0 fps (Kinect ~20 fps).
- IMU span is usually longer than RGB: mean +0.12 s (wrist), +0.18 s (thigh); max 1.15 s.
- Conclusion: clip-level aligned, NOT frame-synchronous. Each modality is resampled
  over its own span. No claims of temporal cross-modal alignment on this dataset.
- Consequence: temporal misalignment must not be used as a semantic conflict operator here.

## Placement confound
- IMU on right wrist for actions 1-21 (671 trials), right thigh for actions 22-27 (190 trials).
- Placement alone separates arm vs leg actions. Semantic swap operators must draw the
  replacement from a different class within the same placement group; cross-group swaps
  are reported only as a separate "easy" condition.

## Sensor limits
- Gyro reaches the +/-1000 deg/s cap in 5/861 trials (<0.01% of values; mostly actions 5, 3, 12).
- Accelerometer reaches the +/-8 g cap in 27 values total.
- Clip-start |acc| median 1.045 g (IQR 1.031-1.083), confirming g units.
- Perturbation operators must clip output to +/-8 g and +/-1000 deg/s.

## Processed tensors (data/processed/utd_mhad_v1.npz)
- X_imu (861, 100, 6), X_skel (861, 40, 20, 3); linear resampling per modality.
- Skeleton: centred on hip_center (joint 4), scaled by mean spine length (per clip, no leakage).
- No z-scoring applied; statistics are fitted per fold on training subjects only.
- Planned RGB length: 32 frames (shortest clip), as frozen features.
