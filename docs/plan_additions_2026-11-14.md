# Plan: three additions before the journal results are frozen again (14 Nov 2026)

Written before any code or result of these experiments. To be reviewed, approved, committed and pushed before
anything runs. Rule (13 Nov): no experiment is skipped. These three were named as "could be added later" in the
approved plans (`plan_nowatch_2026-11-11.md` section 9, `plan_followup_2026-11-12.md` Part A, `plan_faults_2026-11-11.md`
section "not done"); they are now planned properly.

Order of work: **3 -> 2 -> 1** (quickest first). Everything below is run on the frozen systems: fitted on each
fold's validation people with the lie bench only, applied unchanged. Nothing is retrained or refitted.
Revision 1 is the main result everywhere; revision 2 is reported as post hoc, labelled.

**The frozen v2 results do not change.** Each addition is a new, separately named bench with its own output files
(`*_nowatch2*`, `*_imuoff2*`, `*_faults2*`); the existing benches, their files and their tables stay byte-identical
(checked by the reproducibility check at the end). Afterwards: tables, fingerprint, tag `journal-results-frozen-v3`,
reproducibility check.

---

## Addition 3. MMAct: the dead watch as raw zeros (247 real no-watch clips)

### 3.1 Why
On the 247 real clips without a watch, the watch was filled with its training mean (= the simulated removal
`r_fill`). Real devices often send something else when a sensor dies: a stream of zeros. Raw zeros are not the
training mean: the accelerometer then reads 0 m/s^2 on every axis, which is physically impossible while the watch
is worn (gravity alone gives 9.8 m/s^2). So the input is far outside anything seen in training.

### 3.2 Decision D1 (to be confirmed): no "frozen last value" on these clips
The watch was never recorded in these clips, so there is no last value to freeze. Proposed instead: the frozen
(stuck) watch is tested as the stuck-sensor fault of Addition 1, on the normal MMAct clips, where a real recording
exists. These clips get the raw-zeros variant only.

### 3.3 Conditions (3 seeds, the six folds that have no-watch clips)
| Name | Clips | Watch input |
|---|---|---|
| `nw_zero:watch` | the 247 no-watch clips | raw value 0 on all watch channels, then the fold's training normalisation |
| `r_zero:watch` | the same reference clips as before (same person, scene, actions) | the recorded watch replaced by raw 0, same rule |
| `clean`, `nw_fill:watch`, `r_fill:watch` | as in the no-watch bench | unchanged, re-run only to check they reproduce the stored answers exactly |

Systems: confidence alone, C3, REV3 (main), TMC alone, ECML alone, D (dropout model); answers without a certificate:
doing nothing (main model), known missing (backup `b4s_nowatch`).

### 3.4 Main result, named in advance
REV3's use / drop accuracy on `nw_zero:watch` (pooled over the 247 clips), next to doing nothing on the same input
and next to the known-missing backup (84.1%).

### 3.5 Predictions (written before any result)
- N1. Doing nothing is worse with raw zeros than with the mean fill: main-model accuracy on `nw_zero` < 55.9%.
- N2. REV3 flags more clips with raw zeros than with the mean fill (flag share > 91.4%), and names the watch on a
  larger share of flagged clips (> 48.7%).
- N3. REV3's use / drop accuracy on `nw_zero` is higher than doing nothing on `nw_zero`.
- N4. REV3's detection AUROC (no-watch vs reference, mean over the four people with 28 or more clips) is at least
  as high as with the mean fill (0.945).
Reporting rules as in `plan_nowatch_2026-11-11.md` section 7 (per person, pooled, four-person mean; persons 10 and 11
as counts; no interval across people).

### 3.6 Expected machine time
About 5 minutes (the mean-fill no-watch bench took 1.5 min of model runs plus seconds of scoring).

---

## Addition 2. WearGait-PD: IMUs switched off for part of the clip ("partial IMU-off")

### 2.1 Why
The IMU-off test (13 Nov, all predictions confirmed) switched the IMUs off for the whole clip. In the real dropouts
an IMU was missing for 44-79% of the clip on average. This addition asks how the answer changes when the IMUs are
off for only part of the clip.

### 2.2 Bench (WearGait-PD, test part, 10 folds x 3 seeds)
- Conditions `io_part:<group>:k<k>:p<p>`: the first k IMUs of the clip's random order (exactly the order of the
  IMU-off test, `imu_order`) are set to the training mean (0 after normalisation) for one contiguous stretch
  covering **p = 25%, 50% or 75%** of the clip's valid samples; everything else is unchanged.
- The stretch: one random start per (group, clip, p), fully inside the clip, keyed on (group, clip, p) only, so it is
  the same for every k, seed and model. The k chosen IMUs are off for the same stretch.
- k as before: arms 1-2, trunk 1-3, legs 1-8; with three values of p that is 39 conditions plus `clean`.
- p = 100% is the stored whole-clip IMU-off test (not re-run).
- Models run, systems and backups: exactly as in the IMU-off test (`plan_followup_2026-11-12.md` A.3, A.4).

### 2.3 Main result, named in advance
For the legs: REV3 use / drop (revision 1) minus doing nothing, at each k and each p, mean over the 10 folds with
its 95% interval.

### 2.4 Predictions (written before any result)
- I1. For the legs with all 8 IMUs off, REV3's gain over doing nothing grows with p:
  gain(25%) < gain(50%) < gain(75%) < +39.5 points (the stored whole-clip value).
- I2. Doing nothing suffers less when the IMUs are off for part of the clip: legs k = 8 at p = 25%, accuracy at
  least 20 points above the whole-clip value (52.5%).
- I3. REV3's detection AUROC (switched-off vs clean) for the legs at k = 8 grows with p.
- I4. With only one IMU off (k = 1, any group, any p), REV3 does not clearly help: mean gain below +0.25 points.
A prediction that fails is reported as failed.

### 2.5 Expected machine time
About 30 minutes (the whole-clip IMU-off bench took 8 min of model runs for 13 conditions; this has 39).

---

## Addition 1. Two more everyday faults on all three datasets: stuck sensor and saturation

### 1.1 Why
The fault test covered drift, gaps and timing shift. Two other common faults of real sensors are a sensor that
freezes ("stuck") and a sensor whose readings are cut off at its measuring range ("saturation").

### 1.2 Fault definitions (same rules as the existing faults: only valid samples change, padding stays 0,
the valid length never changes, every random draw keyed on (seed, fault, level, clip))
- **`f_stuck`**: from a random onset to the end of the clip, the sensor repeats its value at the onset.
  The frozen stretch covers **25% / 50% / 100%** of the valid samples (100% = the whole clip is one constant value).
  It differs from the existing gap fault, which holds a value in the middle of the clip and then recovers.
- **`f_sat`** (decision D2, to be confirmed): readings clipped at a fixed measuring range, using the standard
  full-scale settings of MEMS motion sensors:
  level 1: accelerometer +/-8 g, gyroscope +/-1000 deg/s; level 2: +/-4 g, +/-500 deg/s; level 3: +/-2 g, +/-250 deg/s.
  Applied per axis in physical units (the units of `src/trustmi/faults.py`). Orientation angles (MMAct phone
  pitch / roll) and the skeleton have no measuring range of this kind and get no saturation.
  The share of samples actually clipped at each level is reported for every sensor (a level can clip almost nothing).
- **Video**: stuck = the frame at the onset repeated to the end (a frozen camera), same 25 / 50 / 100%; saturation =
  over-exposure: every frame's pixel values multiplied by **1.5 / 2 / 3** and clipped to the valid pixel range,
  before the video features are extracted (as the existing video faults, `scripts/extract_video_faults.py`).

### 1.3 Where each fault applies
| Dataset | Sensor | stuck | saturation |
|---|---|---|---|
| UTD-MHAD | IMU (accelerometer, gyroscope) | yes | yes |
| UTD-MHAD | skeleton | yes (tracking freeze) | no |
| UTD-MHAD | video | yes | yes |
| MMAct | watch (accelerometer) | yes | yes |
| MMAct | phone (accelerometer, gyroscope, orientation) | yes | accelerometer and gyroscope only |
| MMAct | video | yes | yes |
| WearGait-PD | arms, trunk, legs (all IMUs of the group) | yes | yes |

### 1.4 Systems, measures and main result
Exactly as in the existing fault test (`plan_faults_2026-11-11.md`): frozen transfer; detection AUROC on harmful
faults (the fault changed the main model's answer) and on all faults; naming the faulty sensor on harmful faults;
fault-clip accuracy before and after use / drop; false alarms on clean clips. The new faults are pooled into their
own group ("new faults") and also reported per fault and level. The existing "all faults" group is not changed.
**Main result, named in advance:** REV3's harmful-fault detection AUROC and naming on the new faults, next to
confidence alone, TMC and ECML, on each dataset.

### 1.5 Predictions (written before any result; based on the pattern of the existing faults on all three datasets)
- F1. Confidence alone is a better alarm than REV3 for harmful new faults on every dataset (higher
  harmful-fault AUROC), as for the existing faults.
- F2. REV3 names the faulty sensor more often than TMC on all three datasets.
- F3. For both new faults, the share of clips whose answer changes grows with the level on every dataset
  (level 1 < level 2 < level 3).
- F4. Saturation level 1 (+/-8 g, +/-1000 deg/s) changes the main model's answer in fewer than 5% of clips on every
  dataset.

### 1.6 Expected machine time
About 2-2.5 h, mostly MMAct video features for the new video faults (the existing three video faults took about
2 h; estimated from file times, so the smoke run is timed first and the estimate corrected before the full run).
Model runs about 35 min, scoring a few minutes.

---

## Checks (all additions)
1. The existing benches are not touched: the lie, fault, no-watch and IMU-off paths re-run on fold 0 seed 0 give
   byte-identical files (as `scripts/check_lies_unchanged_journal.sh` and `scripts/check_unchanged_imuoff.sh`).
2. Each new bench reproduces the stored clean answers exactly before anything is scored.
3. Unit tests for every new input: padding stays 0, valid length unchanged, only the planned sensor changes,
   nested k (Addition 2), raw zero means exactly 0 before normalisation (Addition 3), clipped values never exceed the
   range and unclipped samples are unchanged (Addition 1).
4. Smoke run (fold 0, seed 0) of each bench before its full run.

## Not done
- No retraining, no new thresholds, no change to the main system or to any frozen table.
- No further fault types beyond these two.
