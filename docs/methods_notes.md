# Methodology notes for the journal paper (running log)

Written as we go, in a form that can be moved into the paper's methodology section. Every parameter here is the
one in the code; the file that defines it is named in brackets. Results are NOT recorded here (see the dated
progress notes); only what was done and how.

---

## A. Real sensor faults (frozen transfer test) — 11 Nov 2026
Plan: `docs/plan_faults_2026-11-11.md` (approved before any result). Code: `src/trustmi/faults.py`,
`scripts/extract_video_faults.py`, `scripts/cert_fault_score.py`.

- **Purpose.** Test whether a certificate fitted only on cross-sensor lies transfers, unchanged, to ordinary
  sensor faults it never saw.
- **Faults, three strengths each.**
  - *Drift:* an offset growing linearly from zero at the first valid sample to its maximum at the last, in a
    random 3-D direction, added to one sensor: accelerometer 0.05 / 0.2 / 0.5 g, gyroscope 1 / 5 / 20 deg/s,
    pitch and roll 2 / 10 / 30 deg; physical channels clipped to the training range of the channel.
  - *Gap:* one stretch covering 10 / 25 / 50% of the clip's valid samples, at a random position, in which the
    sensor holds the value just before the stretch (just after it, if the stretch starts the clip); for video,
    3 / 8 / 16 of the 32 decoded frames are frozen before feature extraction.
  - *Timing error:* one sensor shifted by 0.2 / 0.5 / 1.0 s, randomly earlier or later, the uncovered edge
    holding the first or last value (MMAct only); for video, the same shift in frames.
  - Video faults are applied to the decoded frames and passed through the same frozen video network
    (R(2+1)D-18) as the clean videos; one random draw per clip, shared by all camera views.
  - All draws are deterministic, keyed on (seed, fault, strength, clip).
- **Frozen protocol.** Every threshold, naming rule and decision rule is fitted on the validation person with
  the lie bench only and applied unchanged to the fault clips of the test person. One extra system, labelled
  "known faults", is fitted with fault clips among the positives, to show the cost of never seeing faults.
- **Measures.** Harm (accuracy drop, answers changed), detection AUROC (all fault clips; harmful fault clips),
  naming accuracy, and accuracy after the use / drop decision (revision 1; revision 2 as post hoc, labelled).
- **Checks.** Clean rows identical between lie and fault benches; frozen fits reproduce the stored lie results
  exactly; recomputed clean video features match the stored ones (largest difference 0.05 of a feature's spread).

## B. Real missing sensor: MMAct clips without a watch — 11 Nov 2026
Plan: `docs/plan_nowatch_2026-11-11.md` (revised once after review, before any code; implementation notes
added before any result). Code: `MMActNoWatch` in `src/trustmi/data/mmact.py`, `scripts/cert_nowatch_score.py`.

- 247 MMAct clips from whole sessions in which the watch was never recorded (6 people); excluded from all
  training since September and used only here, as an extra test set for the fold of their test person.
- **Conditions.** Known missing (the phone + video backup answers); silent (the watch input set to its
  training mean, i.e. the `r_fill` rule, and the frozen certificate decides); flagged absent (the watch masked
  inside the fused model).
- **Reference clips.** The same person's normal clips of the same scene and actions; one person with no
  same-scene normal clip uses other scenes and is reported separately.
- **Main result named in advance:** accuracy of the main certificate's use / drop decision on the silent clips,
  revision 1, against doing nothing and against the known-missing backup.
- **Reporting.** Per person, pooled over clips, and the plain mean over the four people with at least 28 clips;
  AUROC never pooled across people (different fits); people with fewer than 28 clips as counts.

## C. Third dataset: WearGait-PD — from 11 Nov 2026
Audit: `docs/data_audit_weargait.md` (`scripts/audit_weargait.py`). Plan: `docs/plan_weargait_2026-11-11.md`
(approved before any model was trained on it).

### C.1 Data
- 185 people (100 with Parkinson's disease, 85 controls), 8 laboratory tasks each, 1,473 recordings at 100 Hz
  with 13 synchronised body-worn IMUs.
- Labels: the per-sample `GeneralEvent` annotation of the dataset.

### C.2 Clips (`scripts/build_weargait.py`)
- **Classes (6):** Walk, Standing, TandemWalk, Turn, Sitting, TurnToSit. The six balance stances (eyes
  open / closed, feet together / apart / one foot in front) are merged into Standing, as they differ in protocol,
  not in activity. SitToStand (too few segments of at least 2 s) and unlabeled samples are dropped.
- **Sensors:** the 13 IMUs are grouped into three sensors: arms (left and right wrist, 12 channels), trunk
  (forehead, chest at the xiphoid, lower back, 18 channels), legs (left and right thigh, shank, ankle and foot,
  48 channels). Channels: accelerometer and gyroscope (3 axes each) of every IMU; magnetometer and derived
  channels are not used.
- **Resampling:** a 31-tap Hamming-windowed sinc low-pass filter (cut-off 20 Hz, zero phase), then every second
  sample, giving 50 Hz. Missing samples are bridged by linear interpolation for the filter only and are set back
  to missing afterwards.
- **Clips:** 2 s (100 samples at 50 Hz), step 1 s, lying entirely within one labelled segment; at most 8 clips
  per segment, taken from the segment start, so that long balance and tandem-walking segments do not dominate.
- **Clean vs real-dropout clips:** a clip is clean if no sample of any channel is missing within the clip
  widened by the filter's reach (15 samples at 100 Hz on each side); otherwise it is a real-dropout clip, and the
  sensors with missing samples are recorded. An IMU absent from a whole recording counts as missing.

### C.3 Folds (`scripts/make_splits_weargait.py`, `splits/weargait_v1.json`)
- 10 person-wise folds fixed once with seed 0: the Parkinson's and control people are shuffled separately and
  split into 10 groups each, so every fold holds 10 Parkinson's people and 8 or 9 controls.
- Fold k: test = fold k, validation = fold k+1 (cyclic), training = the remaining 8 folds. Every person is
  tested exactly once. Seeds 0, 1, 2.
- Only clean clips are used for training, validation and testing. Real-dropout clips never enter training or any
  fitting; they form an extra test set for the test people of each fold.

- Built: 36,318 clips (32,735 clean, 3,583 real-dropout). Clean class shares: Standing 38.8%, Walk 37.9%,
  TandemWalk 7.8%, Turn 7.6%, Sitting 5.9%, TurnToSit 2.2%. Four people (2 with Parkinson's disease, 2
  controls) have no clean clip and appear only in the real-dropout test set.

### C.4 Loader (`src/trustmi/data/weargait.py`)
- Per-channel normalisation with mean and standard deviation from the training people's clean clips only.
- In real-dropout clips, missing samples are set to the training mean (zero after normalisation): the same
  `r_fill` rule as for the simulated removals and the MMAct no-watch clips.

### C.5 Verification so far
- Build script: 5 tests on synthetic recordings with known answers (clip boundaries, cap, label merging, filter
  pass-band and stop-band, missing-sample propagation, absent IMUs, determinism); three deliberately planted
  bugs were each caught.
- Folds and loader: tests on the built data (person-wise folds, balance, no clip lost or duplicated, train-only
  statistics, finite inputs, missing samples equal to the training mean).

### C.6 Models (`scripts/train.py`, `scripts/train_mi.py`, `src/trustmi/models/fusion.py`)
- Every model of the other datasets is trained with the same architecture and settings (AdamW, learning rate
  3e-4, weight decay 0.05, cosine schedule, batch 32, at most 60 epochs, early stopping on validation macro-F1
  with patience 15, bf16, gradient clipping at 1.0); only the input sizes differ. Each sensor group uses the
  same convolutional IMU encoder as the MMAct devices (stride 4: 100 samples -> 25 tokens).
- Models: three single-sensor witnesses; the fused model over the three groups (main); three backups (fused
  without one group); TMC and ECML over the three groups; the fused model trained with 50% sensor dropout; a
  shuffled-label control of the fused model (seed 0). Three seeds per fold.
- MI matchers as for MMAct (InfoNCE over every sensor pair, 16 wrong partners from the same person,
  1,000 anchors per epoch). Wrong partners and swap donors never come from the same labelled segment as the
  anchor, since clips of one segment overlap in time.
- Backups for the drop action chosen by the existing rule (validation accuracy, 1-point margin in favour of the
  single-sensor model).

### C.7 Test benches (`scripts/run_journal_queue_i.sh`)
- **Unit of analysis:** the fold (about 18 test people). The certificate is fitted on the fold's validation
  people and applied to its test people; results are averaged over the 10 folds, with 95% intervals over folds.
- **Lies** as for the other datasets: each sensor group replaced by the same person's recording of a different
  activity (lie) or of the same activity (control), and a harmless level-2 change. Donor clips never come from
  the same labelled segment as the target clip (clips of one segment overlap in time).
- **Harmless changes on a group:** every IMU gets its own random mounting rotation (the IMUs are mounted
  separately); gain and jitter per channel; the physical range of each channel is that of the training people.
- **Faults:** drift, gap and timing error, three strengths each, as for MMAct, applied to all IMUs of one group;
  under drift every IMU of the group drifts in its own random direction.
- **Real dropouts:** clips in which one or more groups really lost samples (missing samples set to the training
  mean), scored frozen against the fold's clean clips; the main result is the use / drop accuracy of the main
  certificate on clips with exactly one group missing, against doing nothing and against the backup for that
  group. Clips with two or three groups missing are reported separately.
