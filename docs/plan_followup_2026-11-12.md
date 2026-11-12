# Plan: follow-up experiments before the journal results are frozen (12 Nov 2026)

Fixed before any code or result of these experiments. To be approved, committed and pushed before anything runs.

## 0. Decisions (12 Nov, taken by Deepinder; to be shown to the supervisor with the manuscript)
The three open questions of progress note 2026-11-12, section 10, are answered as suggested there:
1. The post hoc explanation of the WearGait-PD real-dropout result (progress note 2026-11-12, section 9.5) goes
   into the paper, labelled "found after the main result, descriptive only".
2. That explanation is turned into a planned test (Part A below), with the predictions written down first.
3. Both rule changes noticed earlier are scored (Part B), but only as clearly labelled post hoc extra rows,
   never as the main result, because both were thought of after seeing results.

Then results are frozen (Part C) and the paper is drafted over the weekend (14-15 Nov).

---

## Part A. Planned test: how many IMUs of one body group are switched off ("IMU-off test")

### A.1 Why
On WearGait-PD the named main result for real dropouts was a near tie (+0.7 points, REV3 better in 2 of 10 folds).
Looking afterwards, REV3 helped a lot when most IMUs of a group were missing (6 or 7 of the 8 leg IMUs) and cost a
little when only 1 or 2 were missing. That look rests mostly on one person (WHC003, fold 2) and a few others, and
was made after seeing the result. This test asks the same question on every test person, with a controlled design.

### A.2 Question
How does the benefit of REV3's use / drop decision change with the number of IMUs switched off in one body group?

### A.3 Test bench (WearGait-PD only; the only dataset where one "sensor" is several IMUs)
- Clips: the clean test clips of every fold (exactly the `clean` rows of the real-dropout bench).
- Conditions, one per group and number k:
  - `io_fill:arms:k1`, `k2` (2 wrist IMUs)
  - `io_fill:trunk:k1` to `k3` (forehead, chest, lower back)
  - `io_fill:legs:k1` to `k8` (thighs, shanks, ankles, feet)
  plus `clean`. 13 conditions plus clean.
- Switching off: all 6 channels (accelerometer and gyroscope) of each chosen IMU are set to the training mean
  (0 after normalisation) for the whole clip. This is the same fill rule as the simulated removal `r_fill` and as
  the real dropouts. Every other IMU and every other group is left exactly as it is.
- Which IMUs: for each clip and group, one random order of that group's IMUs is drawn (keyed on group and clip,
  the same for every seed and model); condition k switches off the first k IMUs of that order. So k+1 is always
  k plus one more IMU of the same clip ("nested"), and across clips every IMU is switched off equally often.
- k = all IMUs of a group is by construction identical to `r_fill:<group>` of the real-dropout bench (check A.7).
- Whole clip, not part of it: the question is how many IMUs, not for how long. This differs from the real
  dropouts, where an IMU was missing for 44-79% of the clip on average. Stated as a limit.
- Models run on the bench: the same as for the real-dropout bench (three witnesses, the main model, the three
  backups, the own MI matcher, TMC, ECML, the 50% dropout model), plus the main model with the whole group flagged
  absent (M), for reference. No training.

### A.4 Systems and protocol
Frozen, exactly as for the faults, the no-watch clips and the real dropouts: every system is fitted on each fold's
validation people with the lie bench only and applied unchanged. Nothing is refitted. Systems: confidence alone,
C3, REV3 (main), TMC alone, ECML alone, D (dropout model). Revision 1 is the main result; revision 2 is reported as
post hoc, labelled. Answers without a certificate: doing nothing (main model), known missing (the backup for that
whole group, `configs/backups.json`), M (group flagged absent), the dropout model with the group hidden.
Unit: the fold (about 18 test people), seeds pooled within a fold; 95% intervals over the 10 folds.

### A.5 Main result, named in advance
For the legs: the paired difference in accuracy, **REV3 use / drop (revision 1) minus doing nothing**, at each
k = 1 to 8, mean over the 10 folds with its 95% interval.

### A.6 Predictions, written before any result
These come from the post hoc look (progress note 2026-11-12, section 9.5). The test checks them on every person.
Each gets a plain verdict, "confirmed" or "not confirmed", whatever comes out.
- **P1 (few leg IMUs off):** at k = 1 and at k = 2 the mean difference is between -2 and 0 points
  (REV3 loses a little, or nothing).
- **P2 (most leg IMUs off):** at k = 7 the mean difference is above 0 with the whole 95% interval above 0.
  (k = 8 is the already known `r_fill` endpoint, 52.5% vs 92.0% pooled; it is shown but is not new evidence.)
- **P3 (trend):** the difference grows with k. Per fold, the least-squares slope of the difference over
  k = 1 to 8 is positive in at least 9 of 10 folds, and the 95% interval of the mean slope is above 0.
- **P4 (crossover):** the crossover k* = the smallest k such that the mean difference is >= 0 at k and at every
  larger k; predicted to lie between 3 and 6 (inclusive).
- **P5 (arms and trunk):** for the arms (k = 1, 2) and the trunk (k = 1, 2, 3) the mean difference stays within
  +/- 1.5 points at every k (the main model depends little on these groups, and their backup is the strong legs
  witness).

### A.7 Supporting measurements and checks
- Per group and k: accuracy of doing nothing, of each system's use / drop, of known missing, M and the dropout
  model hidden; detection AUROC (IMU-off clips vs the same fold's clean clips); share flagged; naming (the group
  with IMUs off is the right answer).
- Checks before any number: (1) clean rows identical to the lie bench; (2) every frozen fit reproduces the stored
  lie numbers exactly; (3) the k = all rows equal the `r_fill:<group>` rows of the real-dropout bench, column by
  column; (4) bench tests: exactly k IMUs (6k channels) set to 0, everything else byte-identical, the same output
  on a second call, the subsets nested, each IMU switched off about equally often.
- Lie, fault and real-dropout results unchanged by the new option (fold 0 seed 0 compared row by row with the
  stored files, as before).

### A.8 Cost
About 20-30 min on the GPU (the real-dropout bench took 5 min with 4 conditions; this one has 14).

---

## Part B. Two rule changes, post hoc (labelled extra rows only)

Both were designed after seeing results. They are scored next to REV3 revision 1, never instead of it, and every
table that shows them says "post hoc". They change nothing in the main system or in any earlier number.

### B.1 Rule (a): confidence as a second alarm
- Idea (from the fault test): ordinary faults that break the answer make the model unsure, which plain confidence
  catches better than REV3; lies keep the model confident, which REV3 catches. Use both as alarms.
- Alarm score = the larger of two percentiles: REV3's score D and the low-confidence score 1 - msp, each turned
  into a percentile among the fold's **validation clean clips** (mid-rank; ties count half).
- Used for detection (AUROC) and for abstaining ("answer the safest 80%"). The use / drop decision stays exactly
  revision 1's (REV3's score, threshold and naming).
- Nothing is fitted on faults or missing data.

### B.2 Rule (b): drop only when sure which sensor is bad
- Idea (from the MMAct no-watch clips and the WearGait-PD real dropouts): a drop with the wrong name, or of a group
  that is still mostly working, costs accuracy.
- Naming margin: for the "odd one out" rule, the named sensor's divergence minus the second largest; for the
  "lowest MI share" rule, the second lowest MI share minus the lowest. (Whichever rule the fold chose on
  validation, as now.)
- Drop when D >= t **and** margin >= tau; otherwise keep the main model's answer. t is revision 1's threshold,
  unchanged.
- tau is chosen on the fold's validation people with the lie bench only, from {0} and the 5%, 10%, ..., 95%
  quantiles of the margin among validation clips with D >= t, to give the highest validation accuracy; ties go to
  the smallest tau (closest to revision 1). tau = 0 means the rule equals revision 1 for that fold; how often that
  happens is reported.

### B.3 Rows and benches
- Three extra rows: REV3 + (a), REV3 + (b), REV3 + (a) + (b). REV3 (main model, revision 1) only.
- Benches: lies (UTD-MHAD, MMAct, WearGait-PD), faults (all three), MMAct no-watch clips, WearGait-PD real
  dropouts, and the IMU-off test of Part A.
- Measures: those of each bench's existing table (detection AUROC, accuracy after use / drop, answer the safest
  80%, flagged share).
- Check: with tau fixed at 0 and the alarm switched off, the new code reproduces REV3 revision 1 on every bench
  exactly (difference 0).

### B.4 What each rule is meant to fix (expectations, not tests)
- (a) should raise fault detection towards confidence's level, possibly at a small cost on lies.
- (b) should reduce harmful drops on the no-watch clips and on partial dropouts, possibly at a small cost on lies
  when it withholds a correct drop.

---

## Part C. Freeze, tables, reproducibility (13 Nov)
1. After Parts A and B are scored, committed and pushed: git tag `journal-results-frozen`. From then on nothing is
   re-run unless a bug is found; any such re-run is logged with its reason.
2. One script builds every journal table and figure from the saved per-fold / per-person result files (no number
   typed by hand): lies, faults, missing sensors (no-watch, real dropouts, IMU-off), dropout-model comparison,
   post hoc rows, and per-fold / per-person spread figures.
3. Reproducibility check: a fresh clone of the repository at the tag, with the same environment; every scorer and
   the table script re-run from the stored model outputs; all outputs compared with the frozen ones (identical, or
   float differences below 1e-12).

## Not done
- No retraining and no new model. No change to REV3, to revision 1 or to any earlier number.
- No partial-in-time version of the IMU-off test (could be added later, labelled).
- The post hoc rules are not tuned on any bench other than the validation lie bench.

## Schedule
- Thu 12 Nov: this plan; IMU-off bench, tests, smoke run, full run; post hoc scorer written.
- Fri 13 Nov: scoring (A and B), progress note, methods notes, freeze, tables and figures, reproducibility check.
- Sat-Sun 14-15 Nov: manuscript draft; ready Sunday evening for the supervisor.
