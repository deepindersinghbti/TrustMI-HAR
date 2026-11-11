# Plan: the 247 MMAct clips without a watch (11 Nov 2026)

Fixed before any result on these clips. To be approved before anything runs.

## 1. What these clips are
- 247 MMAct clips where the watch was never recorded; phone (accelerometer, gyroscope, orientation) and all
  four camera views are complete (`docs/data_audit_mmact_video.md`, status `no_watch3`).
- They come in whole sessions, from 6 people: person 2 (126 clips, all scene 4), 16 (54), 13 (28), 20 (28),
  11 (6), 10 (5). 28 of the 35 actions appear.
- Decided in September (`splits/mmact_loso_v1.json`): never used for training; used only as an extra test set
  for the fold whose test person they belong to. Video features already exist (extracted 5 Nov).
- So far unused. This is the only real (not simulated) missing sensor in our data.

## 2. Questions
1. When the watch is really missing, how well does each system answer?
2. Does the simulated removal used so far (`r_fill`: sensor set to its training mean; `r_mask`: sensor flagged
   absent) behave like a real missing sensor?

## 3. Conditions per clip (3 seeds, test person's fold only)
| Name | What the system receives | Meaning |
|---|---|---|
| K, known missing | the device reports the watch as absent; the backup model (`b4s_nowatch`, phone + video) answers | upper reference; needs no certificate |
| F, silent, filled | the watch is filled with its training mean (exactly the `r_fill` rule); the full model answers; the frozen certificate decides use / drop / abstain | the realistic silent failure |
| M, flagged absent | the watch is flagged absent (exactly the `r_mask` rule) | as in the lie test |

Note: by construction, F is numerically the same input rule as `r_fill`. Any difference between the real
and the simulated case therefore comes from the clips themselves (other sessions, mostly scene 4), not
from the filling.

## 4. Systems
The same as in the fault test, all frozen (fitted on the validation person with the lie test only):
confidence alone, C3, REV3 (main), TMC, ECML, D (dropout model). No "known" row: nothing is fitted on these
clips. Revision 1 is the main result; revision 2 is reported as post hoc, labelled.

## 5. Measurements
1. **Accuracy:** K; full model alone on F; each system's use / drop policy on F; M.
2. **Flag and naming:** share of clips where the system drops a sensor, and share of those where it names
   the watch (here the watch is the right answer on every clip).
3. **Abstain:** accuracy on the 80% most trusted clips (no-watch clips together with the same people's clean
   clips).
4. **Real vs simulated:** for the same 6 people, compare REV3's score, flag share, naming and policy accuracy
   on the real no-watch clips (F) with `r_fill` of the watch on their normal clips (stored lie-test results).
5. **Are these sessions harder?** Accuracy of K (`b4s_nowatch`) on the no-watch clips vs on the same people's
   normal clips. A gap here means the sessions differ, which must be kept apart from any certificate effect.

## 6. Limits, stated in advance
- Only 6 people, and person 2 holds 51% of the clips. Results are given per person and pooled;
  no 95% interval is claimed across people. Pooled numbers are dominated by person 2 / scene 4.
- These clips are a different set of sessions, not the same clips with the watch removed. Question 2 can
  therefore only show agreement or disagreement, not prove the simulation correct.
- One answer per clip and seed; seeds are averaged as everywhere else.

## 7. Implementation
1. `src/trustmi/data/mmact.py`: a new function `load_nowatch(fold)` (no existing function changed). It returns
   the no-watch clips of the fold's test person: same window, same train-only normalisation, watch = 0 after
   normalisation (= training mean), and video matched by clip name.
2. Tests (`tests/test_nowatch.py`): 247 clips over the 20 folds, each in exactly one fold, none in the main
   set; watch exactly 0; phone and video built exactly like the main clips; trial names match the video file.
3. `scripts/cert_nowatch.py`: runs every model and witness on these clips (conditions K, F, M); first checks
   that the same models still reproduce their stored clean answers on the normal clips.
4. `scripts/cert_nowatch_score.py`: reuses the frozen fit from `cert_fault_score.py`; checks again that the
   frozen fit reproduces the stored lie numbers exactly.
5. Run on the desktop (expected: a few minutes; 247 clips x 3 seeds), then the progress note.

## 8. Not done
- No retraining, no new thresholds, no change to the main system.
- These clips stay out of every existing table; they get their own table.
