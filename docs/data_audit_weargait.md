# Data audit: WearGait-PD (11 Nov 2026)

Purpose: decide whether WearGait-PD can be a third dataset for the journal. Read-only audit; nothing trained.
Script: `scripts/audit_weargait.py` (log: `logs/audit_weargait.log`; tables: `data/processed/weargait_files.parquet`,
`data/processed/weargait_segments.parquet`). All numbers below come from that run.

## 1. What is in it
- **185 people:** 100 with Parkinson's disease (PD), 85 healthy controls.
- **8 lab tasks per person:** Balance, SelfPace, HurriedPace, SelfPace_mat, HurriedPace_mat, SelfPace_matTURN,
  TUG (timed up and go), TandemGait. 1,473 recordings; 3 PD people miss 1-3 tasks (NLS033, NLS056, NLS201).
- **100 Hz** in every file. Recordings last 16-318 s (median 27-150 s depending on task).
- **13 body-worn IMUs** (accelerometer, gyroscope, magnetometer, orientation): forehead, chest (xiphoid),
  lower back, both wrists, thighs, shanks, ankles, feet. Also foot contact / foot pressure and a pressure walkway
  (walkway only on the _mat tasks).
- Also present, not used here: demographic + clinical tables (MDS-UPDRS), walkway gait metrics, and two
  real-world tasks (FreeWalk, mock door frame) for a subset of people.

## 2. Labels
Every time step has a `GeneralEvent` label. Over all files (15,925 segments):

| Label | Segments | Median length (s) | People |
|---|---|---|---|
| Walk | 4,989 | 3.7 | 185 |
| Turn | 3,863 | 2.0 | 185 |
| Standing | 1,747 | 4.2 | 185 |
| Sitting | 741 | 3.6 | 184 |
| SitToStand | 554 | 1.4 | 184 |
| TurnToSit | 552 | 2.6 | 184 |
| TandemWalk | 363 | 12.9 | 182 |
| 6 balance stances (eyes open/closed, feet together/apart/one in front) | 181-184 each | 14-16 | 181-184 |
| unlabeled | 2,016 | 6.1 | 185 |

- Activity labels exist, so the dataset fits activity recognition, not only gait analysis.
- SitToStand is short: median 1.4 s, only 99 segments of 2 s or more.
- The six balance stances are all "standing still"; eyes open vs closed is not expected to be separable from
  body-worn motion sensors. They should be merged into one class.
- A `ClinicalEvent` column marks rare events (freezing of gait, stagger, attempts to rise, use of arm rests).

## 3. Missing sensors (real, not simulated)
- 1,429 of 1,473 files (97%) have all 13 IMUs. 43 files lack 1 to 7 locations; 1 file has no IMU data.
- 10.8% of files have a sensor with more than 1% missing samples; where it happens, 21-37% of that sensor's
  samples are missing. No sensor is missing for a whole file.
- These are **real partial dropouts**, spread over all 13 locations: a natural fault test, like the MMAct
  clips without a watch but on a much larger set of people.

## 4. Fit to our setup
| Need | WearGait-PD |
|---|---|
| Several sensors recorded at once | yes: 13 IMUs, 100 Hz, synchronised in one file |
| A label per clip, several classes | yes: about 7 classes after merging (Walk, Turn, Standing, Sitting, SitToStand, TurnToSit, TandemWalk; balance merged into Standing or kept as one class) |
| Enough people for person-wise testing | yes: 185 (UTD-MHAD 8, MMAct 20) |
| Real sensor failures | yes: partial dropouts in about 11% of files |

Concerns:
- **Fewer, easier classes** than UTD-MHAD (27) and MMAct (35); walking vs sitting is easy, so lies may be
  caught partly because accuracy is very high.
- **The task gives the label away.** Each task contains only 2-5 labels (TandemWalk only in TandemGait,
  sitting only in TUG), so clips must be cut from segments, never whole files, and swaps must come from
  other segments.
- **Short segments:** turns about 2 s, SitToStand about 1.4 s; a fixed window of about 2 s would be needed.
- **PD vs control:** a possible confound (PD people walk differently) but also a useful split for reporting.
- **13 IMUs must be grouped into a few "sensors"** for our method, e.g. wrists / trunk (forehead, chest,
  lower back) / legs (thighs, shanks, ankles, feet). This grouping is a design choice to fix in advance.

## 5. Recommendation
**Usable, and worth adding**, mainly because it brings many people and real sensor dropouts. It is not a quick
add: it needs a loader, windows and splits, then every model the other datasets have (fused model, single-sensor
witnesses, backups, MI matchers, TMC/ECML, dropout model) and all test benches. Rough cost: 1 day of code and
checks, 1-2 days of GPU. With 185 people, leave-one-person-out is too slow; a fixed person-wise split
(e.g. 10 folds) is needed.

## 6. Decisions to fix before any building (not yet taken)
1. Class list (merge the six balance stances; keep or drop SitToStand; drop "unlabeled").
2. Window length and step (about 2 s), and how segments shorter than the window are handled.
3. Sensor grouping (proposal: wrists / trunk / legs = 3 sensors, like the other datasets).
4. Which IMU channels (accelerometer + gyroscope as for MMAct; magnetometer left out).
5. Folds: number of person-wise folds, PD and controls balanced in each.
6. How files and segments with missing sensor data are used: kept out of training and used as the real-dropout
   test set, as for the MMAct no-watch clips.
