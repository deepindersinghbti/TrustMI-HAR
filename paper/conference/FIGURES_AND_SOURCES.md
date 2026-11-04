# TrustMI-HAR conference draft: figures to make, and where every number comes from

## Figures needed (placeholders in the .tex, replace with \includegraphics)

| Fig. | What | Size | How | Data source |
|---|---|---|---|---|
| 1 | Pipeline flowchart: sensors -> fused model and unimodal witnesses in parallel -> three signals -> trust score -> odd-one-out naming -> use / drop / abstain | single column, vertical, about 5 cm tall | draw.io (you) | none |
| 2 | Per-subject lie detection AUROC: certificate vs TMC, ECML, 1-MSP. Two panels (UTD-MHAD 8 subjects, MMAct 20 subjects), one line per subject | single column, about 4.2 cm tall | Python script (next step) | results/certificate/e_d1_per_subject.csv, e_d2_per_subject.csv (revision 1 rows; columns lie_auroc, lie_auroc_msp, system) |
| 3 | Naming matrix, UTD-MHAD: lying sensor (rows) vs named sensor (columns), row %, 3x3 heatmap | single column, about 3.6 cm tall | Python script (next step) | logs/cert_attrib_v3.log ("who gets blamed" table: 86.5/9.8/3.6, 17.0/78.7/4.3, 10.8/5.2/84.0) |

## Number sources

| Paper value | Source file |
|---|---|
| Clean accuracy, witnesses and B4, UTD-MHAD (88.2, 76.8, 54.6, 95.0) | runs/baselines.log (seed-averaged per fold), B4 also results/operators/A3_subject_level_table1.csv |
| Clean accuracy MMAct (74.2, 56.3, 79.0) | results/summaries/baselines_d2.txt |
| TMC / ECML clean accuracy (94.4, 93.5, 77.5, 77.8) | runs/evidential_d1.log, runs/evidential_d2.log; e_d*_per_subject.csv acc_main_normal |
| Answer-change table (63.1, 3.4, 0.5, 73.4, 16.3, 2.9, 59.5, 15.1, 4.0), 35.8% accuracy and 0.668 / 0.881 / 0.851 / 0.931 confidence | results/operators/A3_subject_level_table1.csv (h_all level 2) |
| 48 of 48 cases | results/operators/A3_subject_level_table2.csv (8/8, 20/20, 20/20) |
| Witness disagreement (57.6, 52.4, 98.2 to 99.7, 94.9 to 96.3, 53.2 to 57.8) | logs/experts_d1_allsensors_backups.log, logs/experts_d2.log (SANITY tables) |
| Lie AUROC table, paired differences and subjects better; error AUROC MMAct; naming 83.1 / 74.1 / 67.0 and differences; B4 lie accuracy 60.9, TMC 74.9, ECML 65.3 | results/certificate/e_d1_per_subject.csv, e_d2_per_subject.csv (recomputed, see numbers_raw.txt) |
| Per-sensor detection (0.908/0.791, 0.897/0.788, 0.812/0.605) and naming per sensor (86.5, 78.8, 84.1) | logs/cert_attrib_v3.log |
| Preliminary signals (0.568 to 0.755 vs 0.791; 0.513 to 0.629 vs 0.686) | results/certificate/v2_auroc_summary.csv |
| Policy table and drop rates (93.1, 72.9, 91.2, 94.9, 70.5, 91.9, 18.4%, 6.8%, +12.0 [6.6, 17.4], 90.7) | logs/cert_policy_v4.log, logs/cert_policy_v5.log; e_d1_per_subject.csv for TMC / ECML rows |
| Certificate on TMC table | results/certificate/e_d1_on_tmc_per_subject.csv, e_d2_on_tmc_s_per_subject.csv (recomputed, see numbers_raw.txt) |
| Equivalence to official code (fusion identical, loss within 1.5e-7) | docs/progress_2026-09-25.md, scripts/check_evidential_equivalence.py |
| Dataset facts (861 trials, 8,471 / 247 / 263 clips, 35 classes, preprocessing) | data/README.md, data/processed/utd_mhad_v1.json, data/features/video_r2plus1d18_v1.json |
| Model and training settings | src/trustmi/models/fusion.py, scripts/train.py |
| Harmless perturbation levels, swap donor rules | src/trustmi/operators.py (LEVELS[2]) |
| Certificate signals, logistic regression lambda = 1, naming rule, policy revisions | scripts/cert_score.py, scripts/cert_policy.py, scripts/cert_attrib.py |
| Backup selection rule | configs/backups.json |
