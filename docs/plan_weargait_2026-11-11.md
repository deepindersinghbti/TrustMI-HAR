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
