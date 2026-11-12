# Plan: WearGait-PD as the third dataset (11 Nov 2026)

Fixed before any model is trained on this dataset. To be approved before anything is built.
Basis: `docs/data_audit_weargait.md` and its two tables (`data/processed/weargait_*.parquet`). All counts below
come from those tables.

## 1. Why
- 185 people (100 Parkinson's, 85 controls) instead of 8 or 20: per-person results get much tighter.
- 13 synchronised body IMUs, no camera: tests the certificate on a wearable-only system.
- Real partial sensor dropouts in 199 of 1,473 recordings (68 people): a large real-fault test.

## 2. The six choices

### 2.1 Classes (6)
| Class | From the labels | 2 s windows (cap 8 per segment) | People |
|---|---|---|---|
| Walk | Walk | 37.7% | 185 |
| Standing | Standing + the six balance stances | 39.2% | 185 |
| TandemWalk | TandemWalk | 7.6% | 182 |
| Turn | Turn | 7.6% | 176 |
| Sitting | Sitting | 5.7% | 184 |
| TurnToSit | TurnToSit | 2.2% | 172 |

- The six balance stances are merged into Standing: they differ in eyes open / closed and foot position, which
  is protocol, not activity.
- **SitToStand is dropped:** only 99 segments of 2 s or more, from 56 people.
- "unlabeled" is dropped.
- The classes are unbalanced. Training is NOT changed for this (same settings as the other datasets);
  balanced accuracy is reported next to accuracy.

### 2.2 Windows
- 2 s windows, step 1 s, lying fully inside one labelled segment (a window never crosses a label change).
- At most 8 windows per segment, taken from the start of the segment, so long balance and tandem segments do
  not dominate. About 36,000 windows in total.
- 100 Hz is reduced to 50 Hz (every second sample, after a low-pass filter), as for MMAct: 100 samples per window.
- One window = one clip. Clips of the same segment stay together in one fold (all splits are by person).

### 2.3 Sensors: 13 IMUs in 3 groups
| Sensor | IMUs | Channels |
|---|---|---|
| arms | left and right wrist | 12 |
| trunk | forehead, chest (xiphoid), lower back | 18 |
| legs | left and right thigh, shank, ankle, foot | 48 |

Three sensors, as in the other two datasets, so every part of the method (witnesses, odd-one-out naming,
pairwise MI, backups) carries over unchanged.

### 2.4 Channels
Accelerometer (3) and gyroscope (3) per IMU. Magnetometer is left out (depends on the room), and so are the
derived channels (free acceleration, orientation, velocity / orientation increments).
Normalisation: per channel, from the training people only, as for the other datasets.

### 2.5 Folds
- 10 person-wise folds, each with 10 Parkinson's people and 8 or 9 controls (fixed with seed 0, written to
  `splits/weargait_v1.json` and never changed).
- Fold k: test = fold k, validation = fold k+1 (fold 0 after fold 9), training = the other 8 folds.
- 3 seeds (0, 1, 2), as everywhere else.
- Results are also reported separately for Parkinson's and control people.

### 2.6 Real dropouts
- A window is **clean** if all 13 IMUs exist in its recording and none of its samples is missing.
- Every other window is a **real-dropout window**. These never enter training, validation or any
  fitting; they form a separate test set for the test person of each fold.
- In a real-dropout window, missing samples are set to the training mean (the `r_fill` rule) and the group
  with missing data is recorded as the right answer for naming. A window where more than one group has missing
  data is reported separately.

## 3. What is built (same as for MMAct, no tuning)
1. Loader `src/trustmi/data/weargait.py` and its tests (window rules, no window crossing a label change,
   no person in two folds, normalisation from training people only, dropout windows never in training).
2. Models per fold and seed, with the SAME architecture and training settings as the MMAct sensor models
   (only the input sizes change): the fused model, the three single-sensor witnesses, the three backups
   (fused without one sensor), the MI matchers, TMC and ECML, and the 50% sensor-dropout model.
   Backups for the drop action chosen by the existing rule (validation accuracy, 1-point margin).
3. Benches, exactly as for the other datasets:
   - **lies** (sensor swaps, same-action controls, harmless changes);
   - **faults** (drift, gap, and timing error, all three applicable since the IMUs share one clock);
   - **real dropouts** (section 2.6).
4. Scoring with the existing scorers, frozen as for the other datasets (fitted on the validation people with
   the lie bench only).

## 4. Checks
- Before training: class and window counts per fold printed and saved; every fold must contain every class.
- Smoke run on one fold and one seed before the full run.
- Reproduction checks as for the other datasets (stored clean answers reproduce; frozen fits reproduce the
  lie numbers).
- The other datasets' results must stay bit-identical (`scripts/check_lies_unchanged_journal.sh`).

## 5. Cost
About 1 day of code and checks, then about 330 trainings (11 models x 10 folds x 3 seeds), expected 1 to 2
days of GPU time, then the benches and scoring (a few hours).

## 6. Not done
- No change to the method, the architecture or the training settings.
- No use of the walkway, foot-pressure or clinical data, nor of the real-world tasks (FreeWalk, door frame).
- No Parkinson's-vs-control classification: the people's group is used only for balanced folds and reporting.

## 7. Implementation notes (11 Nov, after building the clips, before any model was trained)
Consequences of the plan meeting the data and the code; nothing measured changes.
1. **Built data** (`logs/build_weargait.log`): 36,318 clips, 32,735 clean and 3,583 real-dropout. Class shares
   of the clean clips: Standing 38.8%, Walk 37.9%, TandemWalk 7.8%, Turn 7.6%, Sitting 5.9%, TurnToSit 2.2%.
   Every fold holds all 6 classes and 10 Parkinson's people (`logs/splits_weargait.log`).
2. **Four people (2 Parkinson's, 2 controls) have no clean clip**, only real-dropout clips. They stay in their
   folds and therefore appear only in the real-dropout test set.
3. **Real-dropout clips are unevenly spread over folds** (12 in fold 7, 678 in fold 6); that test is reported
   pooled over clips and per fold, never as a mean over folds alone.
4. **Overlapping clips.** Clips of one labelled segment overlap by 1 s. They are therefore never swap donors
   (same-action control) or MI wrong partners for each other. This follows from the existing rule that a donor
   or wrong partner is a different recording of the same person; it applies only to datasets with segments.
5. **Swap donors** follow the existing rule (same person, same group); the group is the cohort, which is the same
   for all clips of a person, so any other segment of the same person can be a donor.
6. **Harmless changes on multi-IMU sensors.** Each IMU of a sensor group gets its own random mounting rotation
   (the IMUs are mounted separately); gain and jitter apply per channel, as for the other datasets. No orientation
   channels are used, so the orientation-offset change does not apply.
7. **Models, names:** witnesses `b1_arms`, `b1_trunk`, `b1_legs`; main `b4w`; backups `b4w_noarms`,
   `b4w_notrunk`, `b4w_nolegs`; `tmc_w`, `ecml_w`; sensor-dropout model `b4w_sd50`; shuffled-label control for
   `b4w`; MI matchers (own, the main one, and frozen) with 1,000 anchors per epoch as for MMAct.
   Backups chosen with the existing rule (`scripts/choose_backups_journal.py --datasets d3`).

## 8. Implementation notes for the benches (12 Nov, during training, before any bench result)
1. **Unit of analysis = the fold.** UTD-MHAD and MMAct test one person per fold, so every scorer reports per
   person. WearGait-PD folds test about 18 people; the scorers fit on the fold's validation people and score the
   fold's test people together, labelled "fold00"...; means and 95% intervals are over the 10 folds. For the
   other datasets the label is the person's ID, exactly as before (`cert_score.unit`).
2. **Faults on a group** (`trustmi.faults`): the whole group is faulty, as a lie replaces the whole group.
   Drift: every IMU of the group drifts, each in its own random direction, with the plan's sizes (0.05 / 0.2 /
   0.5 g; 1 / 5 / 20 deg/s). Gap and timing error: all channels of the group together. Timing error applies
   (all IMUs share one clock at 50 Hz after loading). Single-IMU failures are covered by the real-dropout test.
3. **Real-dropout bench** (`cert_conditions.build_realdrop_conditions`, `scripts/cert_realdrop_score.py`):
   - real-dropout clips with exactly one group missing are the main set; two or three groups are reported apart;
   - reference = all clean clips of the fold's test people; simulated removal = `r_fill` of each group on them;
   - answers without a certificate: doing nothing; known missing (the chosen backup for the missing group; for
     two missing groups the witness of the remaining group; none for three); the missing group(s) flagged
     absent in the full model; the dropout model with the missing group hidden;
   - **main result, named in advance:** REV3's use / drop accuracy, revision 1, on the single-group
     real-dropout clips, pooled over clips, against doing nothing and against known missing; also per fold;
   - detection AUROC (single-group real-dropout clips vs clean clips) per fold with at least 20 such clips,
     mean over those folds; never pooled across folds.
4. **Dropout model:** only the 50% model exists for this dataset, so the dropout ablation has rows A, B, C(50),
   D(50) only.
5. **Backups are chosen before the benches** (`choose_backups_journal.py --datasets d3`), so
   `cert_signals.py` was made to skip backups for every journal dataset, as its own comment already stated
   (backup answers come from `cert_experts.py --extra-models`). The stored MMAct and UTD-MHAD signal files have
   no backup columns, so nothing in them changes.
