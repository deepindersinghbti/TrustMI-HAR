# Plan: real sensor faults (drift, partial dropout, timing error), 11 Nov 2026

**Status: DRAFT, awaiting approval. No code written or run yet.** Once approved, this file is committed
unchanged before any fault result exists, so every definition, strength and metric below is fixed in advance.

## 1. Question

So far every "lie" is a swap: one sensor shows a different action of the same person. A reviewer will ask:
does the certificate also catch the faults that real sensors have, which it has never seen?

The main test is therefore **transfer**: the certificate (REV3), its weights, the use / drop threshold and the
naming rule stay exactly as fitted on swaps. Nothing is refitted and nothing is retrained.

## 2. What the existing code gives us

- `src/trustmi/operators.py` already does the hard parts: converts a sensor to physical units with the fold's
  training statistics, knows which channels are accelerometer / gyroscope / orientation angles, clips to the
  physical range seen in training, keys every random draw on (seed, operator, level, clip), and never touches
  padding. The faults reuse all of this.
- **Not edited:** `operators.py` and every conference code path stay as they are. The faults live in a new
  module that builds on the operator bank, so the conference numbers cannot change
  (`scripts/check_conference_unchanged.sh` is run after any edit, as before).

## 3. The three faults

Common rules (same as the swaps): only valid samples change, padding stays 0, the valid length never changes
(so the padding pattern cannot give a fault away), every draw is deterministic and keyed on the clip, and the
result is clipped to the physical range seen in training (a real sensor saturates). Three strengths each,
from realistic to gross; strengths are **swept, never tuned**.

### 3.1 Drift (slowly growing offset)
A bias that grows linearly from 0 at the first valid sample to its full value at the last one. For a 3-axis
sensor the bias points in one random direction (one draw per clip), as a real bias vector would.

| Channel | Level 1 | Level 2 | Level 3 | Why these values |
|---|---|---|---|---|
| Accelerometer | 0.05 g | 0.2 g | 0.5 g | MEMS zero-g offset and its temperature drift are typically up to about 0.05-0.1 g; 0.5 g is a gross fault (lost calibration, loose mount) |
| Gyroscope | 1 deg/s | 5 deg/s | 20 deg/s | typical MEMS zero-rate offset is about 1-5 deg/s; 20 deg/s is gross |
| Phone pitch / roll | 2 deg | 10 deg | 30 deg | orientation estimated by integrating a drifting gyroscope |

In each dataset's own units (training spread, fold 0, for scale only):
- UTD-MHAD IMU: accelerometer in g (spread 0.53-0.75 g), gyroscope in deg/s (spread 102-125 deg/s).
- MMAct watch and phone accelerometer in m/s^2 (0.05 g = 0.49 m/s^2; spread 3.1-8.6 m/s^2);
  phone gyroscope in rad/s (5 deg/s = 0.087 rad/s; spread 0.64-1.04 rad/s).
- So level 3 drift ends at about 0.6-1.0 training spreads for the accelerometers and about 0.2-0.4 for the
  gyroscopes. The script prints these ratios for every fold.

### 3.2 Partial dropout (a stretch of the clip is lost)
One contiguous gap covering **10% / 25% / 50%** of the valid samples, at a random position fully inside the clip.
The gap is filled by **holding the last value before it** (what a streaming or logging pipeline does when
packets are lost; a gap at the very start holds the first value after it). Whole-sensor loss is already covered
by the existing removal operators.

### 3.3 Timing error (one sensor out of step with the others)
The sensor is shifted by **0.2 s / 0.5 s / 1.0 s** (10 / 25 / 50 samples at 50 Hz), earlier or later at random.
Samples pushed out are lost; the vacated end holds the edge value; the valid length is unchanged.

## 4. Where each fault applies

| Sensor | Drift | Partial dropout | Timing error |
|---|---|---|---|
| UTD-MHAD IMU | yes | yes | no (see below) |
| UTD-MHAD skeleton | no (Kinect has no drift mechanism) | yes (tracking loss: last frame held) | no |
| UTD-MHAD video | no | yes (see section 5) | no |
| MMAct watch | yes | yes | yes |
| MMAct phone | yes (accel, gyro, angles) | yes | yes |
| MMAct video | no | yes (see section 5) | yes (see section 5) |

**Why no timing errors on UTD-MHAD:** our own data audit (`docs/data_audit_utd-mhad.md`) found its sensors are
aligned per clip, not frame by frame. The IMU runs on average 0.12-0.18 s longer than the video, by up to
1.15 s, and every modality is resampled over its own span. A shift of the same size is already part of normal
UTD-MHAD data, so it would not be a fault. MMAct sensors and video cover the same span (audit, section 3),
so a shift there is a real fault.

Counts: UTD-MHAD 3 x 3 + ... = IMU 6 conditions, skeleton 3, video 3 = **12 fault conditions**;
MMAct watch 9, phone 9, video 6 = **24 fault conditions**.

## 5. Video faults (decision needed)

A video sensor is 4 feature tokens from a frozen network, each summarising 8 of 32 frames. Two ways:

- **A. Frame level (recommended).** Apply the fault to the decoded frames (dropout: frozen frame held over
  10 / 25 / 50% of the 32 frames; timing: frames shifted by 0.2 / 0.5 / 1.0 s at the video's frame rate), then run
  the same frozen R(2+1)D-18 to get new tokens. Each video is decoded once and all fault versions are built
  from the same frames. Faithful to what a real camera fault does. Cost: like the 5 Nov extraction
  (MMAct about 70 minutes, UTD-MHAD 15 seconds), plus a few extra network passes per video; about 1.5-3 hours,
  so overnight. Check: the clean version built by this script must equal the stored features.
- **B. Feature level.** Hold or shift whole tokens. Instant, but only in steps of 25% of the clip, and a held
  token is not what the network would output for a frozen camera. Weaker for a journal.

## 6. What is measured (per test person, 3 seeds pooled, mean [95% interval], as before)

For each fault type, sensor and level, on the test person:
1. **Harm:** accuracy of the fused model under the fault, compared with the same clips clean.
2. **Detection:** AUROC of the frozen certificate score, fault clips against clean clips of the same person.
   Also on **harmful faults only** (clips whose fused answer changes because of the fault), because a fault
   that does not change the answer does not need catching. Plus the share of fault clips that cross the
   frozen drop threshold, against the share of clean clips (false alarms).
3. **Naming:** how often the faulty sensor is the one named (all fault clips, and harmful ones).
4. **Recovery:** accuracy on fault clips, fused model alone against the use / drop policy (revisions 1 and 2),
   and "answer the safest 80%".

Systems compared, all frozen as fitted on swaps: confidence alone (1 - msp), C3, **REV3 (main)**, TMC and ECML
(their drop = their own fusion without the named sensor), and the 50% sensor-dropout model as row D.

**Extra row, labelled "known faults":** REV3 with its score refitted on the validation person with fault clips
added to the positives. It answers how much is lost by never having seen a fault. It is reported next to,
never instead of, the frozen result.

## 7. Implementation, in order (each step checked before the next)

1. `src/trustmi/faults.py`: the three faults, built on the operator bank. `tests/test_faults.py`: known-answer
   tests (padding and valid length unchanged, deterministic, drift is an exact linear ramp of the stated size,
   the gap covers exactly the stated share, the shift is exactly the stated number of samples, clipping reported).
2. `cert_conditions.py`: a NEW function `build_fault_conditions` (the existing one is not changed).
3. The test-bench scripts (`cert_experts.py`, `cert_mi.py`, `cert_evidential.py`, `cert_dropout.py`) get a
   `--conditions faults` option that writes separate `_faults` files. Without it, their output is checked to
   be identical to now (conference check + journal fold 0).
4. Video fault features (option A): `scripts/extract_video_faults.py`, overnight.
5. `scripts/cert_fault_score.py`: section 6, and `scripts/run_journal_queue_f.sh`: smoke run first, then full run.
6. Progress note with the chronology, as every day.

## 8. Not in this plan
- No retraining of any model (that would be a different question: training on faults).
- No new fault types beyond the three agreed (stuck sensor and saturation are possible later additions).
- The 247 MMAct clips without a watch and WearGait-PD stay separate items.
