# Plan: the 247 MMAct clips without a watch (11 Nov 2026, revised after review the same day)

Fixed before any result on these clips. To be approved before anything runs.
Revised once after an internal review of the first draft, still before any code or result existed.

## 1. What these clips are
- 247 MMAct clips where the watch was never recorded; phone (accelerometer, gyroscope, orientation) and all
  four camera views are complete (`docs/data_audit_mmact_video.md`, status `no_watch3`).
- They come in whole sessions, from 6 people: person 2 (126 clips, all scene 4), 16 (54, scene 1),
  13 (28, scene 1), 20 (28, scene 4), 11 (6, scene 2), 10 (5, scene 1). 28 of the 35 actions appear.
- All 247 pass the video / sensor length rule of the main clips (ratio 0.9 to 1.1; observed 1.002 to 1.062).
- Decided in September (`splits/mmact_loso_v1.json`): never used for training; used only as an extra test set
  for the fold whose test person they belong to. Video features already exist (extracted 5 Nov).
- So far unused. This is the only real (not simulated) missing sensor in our data.

## 2. Questions
1. When the watch is really missing, how well does each system answer?
2. Do the removal results obtained so far hold on sessions the lie test never touched?
   (Not "does the simulation match reality": the silent-filled input below is, by construction, the same rule
   as the simulated removal `r_fill`. The only thing that differs is the sessions.)

## 3. Conditions and which system gets which (3 seeds, test person's fold only)
| Name | What the system receives | Systems |
|---|---|---|
| K, known missing | the device reports the watch as absent | backup model `b4s_nowatch` (phone + video); dropout model (sd50) with the watch hidden |
| F, silent, filled | the watch is filled with its training mean (exactly the `r_fill` rule); the frozen system decides use / drop / abstain | confidence alone, C3, REV3 (main), TMC, ECML, D (dropout model) |
| M, flagged absent | the watch is flagged absent (exactly the `r_mask` rule) | standard fused model; dropout model (sd50) |

All systems are frozen: fitted on the validation person with the lie test only. No "known" row: nothing is
fitted on these clips. Revision 1 is the main result; revision 2 is reported as post hoc, labelled.

## 4. Main result (named in advance)
REV3's use / drop policy accuracy on the silent-filled clips (F), revision 1, compared with
(a) the full model doing nothing on F and (b) the known-missing backup (K).
Everything else below is supporting.

## 5. Supporting measurements
1. **Accuracy:** K; full model alone on F; each system's policy on F; M.
2. **Detection:** AUROC of no-watch clips (F) against the reference clips (section 6), for every system
   including confidence alone, as in the lie and fault tests.
3. **Flag and naming:** share of clips where the system drops a sensor, and share of those where it names
   the watch (the watch is the right answer on every clip).
4. **Abstain:** accuracy on the 80% most trusted clips, computed per person on that person's no-watch clips
   together with their reference clips, then averaged over the four people with 28 or more clips.
5. **Held-out sessions (question 2):** REV3's score, flag share, naming and policy accuracy on F, next to the
   stored `r_fill`-of-the-watch results on the reference clips.
6. **Are these sessions harder?** Accuracy of K (`b4s_nowatch`) on the no-watch clips against the reference
   clips. A gap here means the sessions differ and must be kept apart from any certificate effect.

## 6. Reference clips (fixed in advance)
- For each person: their normal (`main3`) clips of the **same scene** and the **same actions** as their
  no-watch clips. This exists for persons 10, 11, 13, 16 and 20.
- **Person 2 has no normal scene-4 clips at all.** Person 2 is therefore reported separately, with the normal
  clips of scenes 1 to 3 (same actions) as reference, labelled "no same-scene reference". No conclusion about
  sessions or simulation is drawn from person 2's reference comparison.

## 7. Reporting rules and limits, stated in advance
- Only 6 people. No 95% interval is claimed across people.
- Three numbers per measurement: per person; pooled over all 247 clips (51% person 2); and the plain average
  over the four people with 28 or more clips (2, 13, 16, 20).
- Persons 10 and 11 (5 and 6 clips) are reported as counts, not percentages.
- These are different sessions, not the same clips with the watch removed. Question 2 can show agreement or
  disagreement with the earlier removal results; it cannot prove the simulation correct.
- Seeds are averaged as everywhere else.

## 8. Implementation
1. `src/trustmi/data/mmact.py`: a new function `load_nowatch(fold)` (no existing function changed). It returns
   the no-watch clips of the fold's test person: same window, same train-only normalisation, watch = 0 after
   normalisation (= training mean), video matched by clip name.
2. Tests (`tests/test_nowatch.py`): 247 clips in total, each in exactly one fold, none in the main set;
   watch exactly 0; **no NaN in any model input** (the stored watch columns of these clips are NaN);
   phone and video built exactly like the main clips; clip names match the video file.
3. `scripts/cert_nowatch.py`: runs every model and witness on these clips (conditions K, F, M); first checks
   that the same models still reproduce their stored clean answers on the normal clips.
4. `scripts/cert_nowatch_score.py`: reuses the frozen fit from `cert_fault_score.py`; checks again that the
   frozen fit reproduces the stored lie numbers exactly.
5. Run on the desktop (expected: a few minutes), then the progress note.

## 9. Not done
- No retraining, no new thresholds, no change to the main system.
- These clips stay out of every existing table; they get their own table.
- No "raw zeros" or "frozen value" variant of the dead watch; could be added later as a labelled extra.
