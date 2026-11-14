# Journal numbers (built by scripts/make_journal_tables.py; do not edit by hand)

Mean over test units, 95% interval over units in brackets; 'better' = units where the paired difference is above 0. Revision 1 unless labelled.

## T1_models: Clean test accuracy of every trained model (folds x 3 seeds; seeds averaged within a fold, interval over folds)

Macro-F1 as in scripts/train.py: the mean F1 over the activities present in the test set of that fold.

| Dataset     | Model                                     | Runs | Accuracy %        | Macro-F1             |
|-------------|-------------------------------------------|------|-------------------|----------------------|
| UTD-MHAD    | Main fused model (IMU + skeleton + video) | 24   | 94.7 [92.1, 97.3] | 0.939 [0.907, 0.970] |
| UTD-MHAD    | IMU alone                                 | 24   | 88.2 [85.4, 91.0] | 0.863 [0.829, 0.897] |
| UTD-MHAD    | Skeleton alone                            | 24   | 76.8 [70.5, 83.1] | 0.740 [0.671, 0.809] |
| UTD-MHAD    | Video alone                               | 24   | 52.0 [41.4, 62.6] | 0.461 [0.346, 0.575] |
| UTD-MHAD    | Without IMU                               | 24   | 74.9 [70.2, 79.6] | 0.715 [0.661, 0.769] |
| UTD-MHAD    | Without skeleton                          | 24   | 89.8 [88.0, 91.7] | 0.881 [0.856, 0.906] |
| UTD-MHAD    | Without video                             | 24   | 94.2 [91.9, 96.4] | 0.933 [0.907, 0.959] |
| UTD-MHAD    | Dropout-trained (p=0.5)                   | 24   | 94.0 [91.1, 96.8] | 0.930 [0.897, 0.964] |
| UTD-MHAD    | TMC                                       | 24   | 94.6 [91.8, 97.5] | 0.939 [0.905, 0.972] |
| UTD-MHAD    | ECML                                      | 24   | 92.5 [90.0, 94.9] | 0.912 [0.883, 0.940] |
| MMAct       | Main fused model (watch + phone + video)  | 60   | 89.1 [86.9, 91.3] | 0.890 [0.868, 0.912] |
| MMAct       | Watch alone                               | 60   | 74.0 [70.8, 77.2] | 0.727 [0.696, 0.759] |
| MMAct       | Phone alone                               | 60   | 56.2 [51.2, 61.2] | 0.491 [0.444, 0.539] |
| MMAct       | Video alone                               | 60   | 82.3 [80.2, 84.5] | 0.828 [0.806, 0.850] |
| MMAct       | Without watch                             | 60   | 85.2 [83.4, 87.0] | 0.851 [0.832, 0.869] |
| MMAct       | Without phone                             | 60   | 88.4 [86.4, 90.4] | 0.883 [0.865, 0.902] |
| MMAct       | Without video                             | 60   | 78.5 [74.2, 82.8] | 0.783 [0.742, 0.824] |
| MMAct       | Dropout-trained (p=0.5)                   | 60   | 89.6 [87.4, 91.8] | 0.896 [0.875, 0.917] |
| MMAct       | TMC                                       | 60   | 89.0 [86.9, 91.0] | 0.888 [0.866, 0.910] |
| MMAct       | ECML                                      | 60   | 88.2 [86.4, 90.0] | 0.877 [0.859, 0.896] |
| WearGait-PD | Main fused model (arms + trunk + legs)    | 30   | 96.6 [95.1, 98.1] | 0.935 [0.909, 0.960] |
| WearGait-PD | Arms alone                                | 30   | 88.5 [86.4, 90.6] | 0.804 [0.772, 0.836] |
| WearGait-PD | Trunk alone                               | 30   | 91.7 [90.1, 93.3] | 0.861 [0.834, 0.889] |
| WearGait-PD | Legs alone                                | 30   | 96.1 [94.4, 97.8] | 0.918 [0.891, 0.946] |
| WearGait-PD | Without arms                              | 30   | 96.3 [94.6, 98.0] | 0.929 [0.902, 0.956] |
| WearGait-PD | Without trunk                             | 30   | 95.9 [94.0, 97.8] | 0.917 [0.889, 0.946] |
| WearGait-PD | Without legs                              | 30   | 92.9 [91.1, 94.7] | 0.880 [0.849, 0.910] |
| WearGait-PD | Dropout-trained (p=0.5)                   | 30   | 96.9 [95.5, 98.3] | 0.938 [0.914, 0.962] |
| WearGait-PD | TMC                                       | 30   | 96.6 [95.3, 98.0] | 0.933 [0.909, 0.957] |
| WearGait-PD | ECML                                      | 30   | 96.4 [95.0, 97.9] | 0.926 [0.900, 0.951] |

## T2_lies: Planted lies: detection, naming and accuracy after use / drop (certificate on the main model, revision 1)

Lie AUROC and naming with 95% intervals over units (UTD-MHAD 8 people, MMAct 20 people, WearGait-PD 10 folds).

| Dataset     | System                        | Lie AUROC            | Names the lying sensor % | Accuracy on lie clips % (after use/drop) | Accuracy on normal clips % | Error AUROC (natural mistakes) |
|-------------|-------------------------------|----------------------|--------------------------|------------------------------------------|----------------------------|--------------------------------|
| UTD-MHAD    | C3 (conference certificate)   | 0.871 [0.837, 0.906] | 82.7 [77.3, 88.2]        | 74.0                                     | 90.2                       | 0.912                          |
| UTD-MHAD    | MI only                       | 0.869 [0.826, 0.913] | 81.6 [76.6, 86.5]        | 76.7                                     | 92.6                       | 0.668                          |
| UTD-MHAD    | Stability + MI (no witnesses) | 0.874 [0.831, 0.917] | 81.6 [76.6, 86.5]        | 76.8                                     | 93.1                       | 0.896                          |
| UTD-MHAD    | REV3 (ours, main)             | 0.901 [0.856, 0.945] | 82.2 [76.4, 88.1]        | 75.7                                     | 91.0                       | 0.902                          |
| UTD-MHAD    | REV3, frozen MI encoders      | 0.887 [0.849, 0.926] | 83.2 [78.0, 88.4]        | 74.3                                     | 89.8                       | 0.913                          |
| UTD-MHAD    | TMC                           | 0.827 [0.789, 0.865] | 73.6 [68.7, 78.5]        | 74.8                                     | 94.6                       | 0.849                          |
| UTD-MHAD    | ECML                          | 0.827 [0.799, 0.856] | 69.4 [65.2, 73.6]        | 70.3                                     | 92.3                       | 0.897                          |
| MMAct       | C3 (conference certificate)   | 0.850 [0.828, 0.871] | 84.7 [82.4, 86.9]        | 75.0                                     | 87.6                       | 0.846                          |
| MMAct       | MI only                       | 0.917 [0.899, 0.935] | 86.3 [83.9, 88.6]        | 74.8                                     | 87.5                       | 0.511                          |
| MMAct       | Stability + MI (no witnesses) | 0.918 [0.899, 0.936] | 86.3 [83.9, 88.6]        | 74.9                                     | 87.6                       | 0.751                          |
| MMAct       | REV3 (ours, main)             | 0.930 [0.911, 0.948] | 86.0 [83.6, 88.4]        | 75.5                                     | 87.2                       | 0.829                          |
| MMAct       | REV3, frozen MI encoders      | 0.881 [0.859, 0.903] | 84.7 [82.4, 86.9]        | 74.9                                     | 87.5                       | 0.841                          |
| MMAct       | TMC                           | 0.756 [0.734, 0.777] | 61.9 [60.0, 63.8]        | 64.0                                     | 88.9                       | 0.889                          |
| MMAct       | ECML                          | 0.736 [0.720, 0.753] | 60.9 [59.5, 62.2]        | 61.9                                     | 88.2                       | 0.868                          |
| WearGait-PD | C3 (conference certificate)   | 0.965 [0.955, 0.975] | 94.4 [93.2, 95.5]        | 92.0                                     | 96.2                       | 0.931                          |
| WearGait-PD | MI only                       | 0.912 [0.908, 0.916] | 97.7 [97.2, 98.1]        | 93.5                                     | 96.4                       | 0.532                          |
| WearGait-PD | Stability + MI (no witnesses) | 0.915 [0.911, 0.919] | 97.7 [97.2, 98.1]        | 93.5                                     | 96.6                       | 0.780                          |
| WearGait-PD | REV3 (ours, main)             | 0.980 [0.977, 0.983] | 97.7 [97.2, 98.1]        | 93.5                                     | 96.4                       | 0.922                          |
| WearGait-PD | REV3, frozen MI encoders      | 0.980 [0.975, 0.985] | 95.8 [94.9, 96.7]        | 92.6                                     | 96.4                       | 0.923                          |
| WearGait-PD | TMC                           | 0.947 [0.937, 0.957] | 89.4 [87.3, 91.4]        | 88.6                                     | 96.3                       | 0.919                          |
| WearGait-PD | ECML                          | 0.930 [0.918, 0.941] | 87.9 [85.1, 90.7]        | 88.5                                     | 96.2                       | 0.905                          |

## T2b_lies_paired: Planted lies: REV3 minus each other system, paired per unit (units better / all)

| Dataset     | REV3 minus                  | Lie AUROC                       | Naming (points)              | Lie-clip accuracy (points)   | Normal-clip accuracy (points) |
|-------------|-----------------------------|---------------------------------|------------------------------|------------------------------|-------------------------------|
| UTD-MHAD    | C3 (conference certificate) | +0.029 [+0.003, +0.055] (6/8)   | -0.5 [-2.0, +1.0] (1/8)      | +1.7 [-1.8, +5.2] (4/8)      | +0.8 [-3.2, +4.8] (4/8)       |
| UTD-MHAD    | TMC                         | +0.074 [+0.048, +0.100] (8/8)   | +8.6 [+6.1, +11.2] (8/8)     | +0.9 [-3.5, +5.3] (4/8)      | -3.6 [-7.1, -0.0] (1/8)       |
| UTD-MHAD    | ECML                        | +0.073 [+0.050, +0.097] (8/8)   | +12.8 [+8.7, +17.0] (8/8)    | +5.4 [-1.1, +11.8] (5/8)     | -1.3 [-5.2, +2.5] (4/8)       |
| MMAct       | C3 (conference certificate) | +0.080 [+0.071, +0.089] (20/20) | +1.4 [+0.6, +2.1] (13/20)    | +0.5 [-0.6, +1.6] (9/20)     | -0.4 [-1.2, +0.4] (9/20)      |
| MMAct       | TMC                         | +0.174 [+0.157, +0.191] (20/20) | +24.2 [+22.6, +25.7] (20/20) | +11.5 [+9.2, +13.8] (19/20)  | -1.7 [-3.3, -0.2] (4/20)      |
| MMAct       | ECML                        | +0.193 [+0.177, +0.209] (20/20) | +25.2 [+23.7, +26.6] (20/20) | +13.6 [+11.1, +16.2] (20/20) | -1.0 [-2.9, +0.9] (11/20)     |
| WearGait-PD | C3 (conference certificate) | +0.015 [+0.007, +0.023] (10/10) | +3.3 [+2.4, +4.2] (10/10)    | +1.6 [+1.2, +1.9] (10/10)    | +0.2 [-0.0, +0.5] (7/10)      |
| WearGait-PD | TMC                         | +0.033 [+0.025, +0.040] (10/10) | +8.3 [+6.5, +10.1] (10/10)   | +4.9 [+4.0, +5.8] (10/10)    | +0.1 [-0.3, +0.4] (5/10)      |
| WearGait-PD | ECML                        | +0.050 [+0.041, +0.059] (10/10) | +9.8 [+7.3, +12.3] (10/10)   | +5.0 [+3.4, +6.5] (10/10)    | +0.3 [-0.2, +0.7] (6/10)      |

## T3_lies_on_evidential: The certificate placed on TMC and on ECML (planted lies, revision 1)

| Dataset     | Answering model | System                      | Lie AUROC            | Names the lying sensor % | Accuracy on lie clips % | Accuracy on normal clips % |
|-------------|-----------------|-----------------------------|----------------------|--------------------------|-------------------------|----------------------------|
| UTD-MHAD    | TMC             | TMC                         | 0.827 [0.789, 0.865] | 73.6                     | 74.8                    | 94.6                       |
| UTD-MHAD    | TMC             | C3 (conference certificate) | 0.864 [0.829, 0.899] | 82.7                     | 74.8                    | 94.3                       |
| UTD-MHAD    | TMC             | REV3 (ours, main)           | 0.901 [0.858, 0.945] | 82.2                     | 76.8                    | 94.0                       |
| UTD-MHAD    | ECML            | ECML                        | 0.827 [0.799, 0.856] | 69.4                     | 70.3                    | 92.3                       |
| UTD-MHAD    | ECML            | C3 (conference certificate) | 0.863 [0.829, 0.898] | 82.7                     | 74.5                    | 89.1                       |
| UTD-MHAD    | ECML            | REV3 (ours, main)           | 0.901 [0.857, 0.945] | 82.2                     | 75.5                    | 89.9                       |
| MMAct       | TMC             | TMC                         | 0.756 [0.734, 0.777] | 61.9                     | 64.0                    | 88.9                       |
| MMAct       | TMC             | C3 (conference certificate) | 0.844 [0.824, 0.865] | 84.7                     | 75.0                    | 87.1                       |
| MMAct       | TMC             | REV3 (ours, main)           | 0.929 [0.911, 0.948] | 86.0                     | 74.9                    | 87.7                       |
| MMAct       | ECML            | ECML                        | 0.736 [0.720, 0.753] | 60.9                     | 61.9                    | 88.2                       |
| MMAct       | ECML            | C3 (conference certificate) | 0.845 [0.824, 0.866] | 84.7                     | 75.5                    | 86.8                       |
| MMAct       | ECML            | REV3 (ours, main)           | 0.930 [0.911, 0.948] | 86.0                     | 75.7                    | 86.9                       |
| WearGait-PD | TMC             | TMC                         | 0.947 [0.937, 0.957] | 89.4                     | 88.6                    | 96.3                       |
| WearGait-PD | TMC             | C3 (conference certificate) | 0.963 [0.953, 0.974] | 94.4                     | 92.3                    | 96.4                       |
| WearGait-PD | TMC             | REV3 (ours, main)           | 0.980 [0.977, 0.983] | 97.7                     | 93.9                    | 96.6                       |
| WearGait-PD | ECML            | ECML                        | 0.930 [0.918, 0.941] | 87.9                     | 88.5                    | 96.2                       |
| WearGait-PD | ECML            | C3 (conference certificate) | 0.961 [0.951, 0.972] | 94.4                     | 91.8                    | 96.2                       |
| WearGait-PD | ECML            | REV3 (ours, main)           | 0.981 [0.978, 0.984] | 97.7                     | 93.2                    | 96.3                       |

## T4_naming_d1v2: UTD-MHAD: which sensor REV3 names, by where the lie was planted (% of that sensor's lie clips, mean over people)

| Lie planted in | named IMU % | named skeleton % | named video % | Main model alone right % |
|----------------|-------------|------------------|---------------|--------------------------|
| IMU            | 82.7        | 9.3              | 8.0           | 36.3                     |
| skeleton       | 14.4        | 78.0             | 7.5           | 56.7                     |
| video          | 9.6         | 4.4              | 86.0          | 89.4                     |

## T4_naming_d2v: MMAct: which sensor REV3 names, by where the lie was planted (% of that sensor's lie clips, mean over people)

| Lie planted in | named watch % | named phone % | named video % | Main model alone right % |
|----------------|---------------|---------------|---------------|--------------------------|
| watch          | 83.7          | 5.6           | 10.8          | 66.1                     |
| phone          | 4.9           | 88.8          | 6.3           | 72.6                     |
| video          | 9.0           | 5.4           | 85.6          | 23.9                     |

## T4_naming_d3: WearGait-PD: which sensor REV3 names, by where the lie was planted (% of that sensor's lie clips, mean over folds)

| Lie planted in | named arms % | named trunk % | named legs % | Main model alone right % |
|----------------|--------------|---------------|--------------|--------------------------|
| arms           | 98.6         | 0.5           | 0.9          | 95.2                     |
| trunk          | 1.1          | 97.0          | 2.0          | 91.8                     |
| legs           | 1.1          | 1.5           | 97.4         | 10.7                     |

## T5_faults: Everyday sensor faults (frozen transfer, all faults pooled, revision 1)

Faults = drift, gap (partial dropout) and timing shift, 3 levels each, in every sensor; the certificate is fitted on validation lies only and transferred unchanged ('frozen'). 'REV3, faults in the fit' is a labelled extra row and is not shown here.

| Dataset     | System                      | AUROC, harmful faults | AUROC, all faults | Names faulty sensor % (harmful) | Fault-clip acc. %: alone -> use/drop | False alarms on clean % |
|-------------|-----------------------------|-----------------------|-------------------|---------------------------------|--------------------------------------|-------------------------|
| UTD-MHAD    | Confidence (MSP)            | 0.926 [0.902, 0.950]  | 0.553             | --                              | 90.7 -> 90.7                         | 0.0                     |
| UTD-MHAD    | C3 (conference certificate) | 0.876 [0.838, 0.914]  | 0.557             | 49.9                            | 90.7 -> 86.8                         | 20.7                    |
| UTD-MHAD    | REV3 (ours, main)           | 0.838 [0.791, 0.885]  | 0.563             | 50.5                            | 90.7 -> 87.5                         | 25.0                    |
| UTD-MHAD    | TMC                         | 0.876 [0.846, 0.906]  | 0.549             | 27.1                            | 92.0 -> 92.0                         | 9.5                     |
| UTD-MHAD    | ECML                        | 0.842 [0.777, 0.908]  | 0.553             | 42.7                            | 89.7 -> 89.2                         | 12.1                    |
| UTD-MHAD    | Dropout-trained model       | 0.850 [0.797, 0.902]  | 0.563             | 49.0                            | 90.8 -> 88.9                         | 12.5                    |
| MMAct       | Confidence (MSP)            | 0.898 [0.889, 0.907]  | 0.517             | --                              | 87.6 -> 87.6                         | 0.0                     |
| MMAct       | C3 (conference certificate) | 0.794 [0.775, 0.812]  | 0.530             | 46.1                            | 87.6 -> 86.4                         | 43.8                    |
| MMAct       | REV3 (ours, main)           | 0.774 [0.756, 0.792]  | 0.543             | 47.4                            | 87.6 -> 86.0                         | 30.5                    |
| MMAct       | TMC                         | 0.860 [0.838, 0.882]  | 0.519             | 17.6                            | 87.8 -> 87.8                         | 1.4                     |
| MMAct       | ECML                        | 0.829 [0.809, 0.849]  | 0.517             | 14.0                            | 86.7 -> 86.7                         | 5.1                     |
| MMAct       | Dropout-trained model       | 0.781 [0.763, 0.800]  | 0.542             | 46.4                            | 88.5 -> 87.5                         | 13.2                    |
| WearGait-PD | Confidence (MSP)            | 0.947 [0.933, 0.960]  | 0.512             | --                              | 96.2 -> 96.2                         | 0.0                     |
| WearGait-PD | C3 (conference certificate) | 0.689 [0.655, 0.722]  | 0.526             | 43.0                            | 96.2 -> 95.8                         | 26.4                    |
| WearGait-PD | REV3 (ours, main)           | 0.862 [0.833, 0.890]  | 0.604             | 53.6                            | 96.2 -> 95.9                         | 6.8                     |
| WearGait-PD | TMC                         | 0.931 [0.918, 0.944]  | 0.519             | 44.0                            | 96.3 -> 96.0                         | 4.4                     |
| WearGait-PD | ECML                        | 0.856 [0.784, 0.927]  | 0.521             | 54.7                            | 96.1 -> 95.8                         | 6.3                     |
| WearGait-PD | Dropout-trained model       | 0.880 [0.866, 0.895]  | 0.604             | 50.7                            | 96.6 -> 96.4                         | 2.6                     |

## T5b_faults_paired: Everyday faults: REV3 minus other systems, harmful-fault AUROC, paired per unit

| Dataset     | REV3 minus                  | AUROC, harmful faults           |
|-------------|-----------------------------|---------------------------------|
| UTD-MHAD    | Confidence (MSP)            | -0.088 [-0.127, -0.048] (0/8)   |
| UTD-MHAD    | C3 (conference certificate) | -0.038 [-0.081, +0.006] (3/8)   |
| UTD-MHAD    | TMC                         | -0.038 [-0.084, +0.009] (2/8)   |
| MMAct       | Confidence (MSP)            | -0.124 [-0.140, -0.108] (0/20)  |
| MMAct       | C3 (conference certificate) | -0.019 [-0.031, -0.007] (5/20)  |
| MMAct       | TMC                         | -0.086 [-0.110, -0.063] (1/20)  |
| WearGait-PD | Confidence (MSP)            | -0.085 [-0.120, -0.050] (0/10)  |
| WearGait-PD | C3 (conference certificate) | +0.173 [+0.145, +0.201] (10/10) |
| WearGait-PD | TMC                         | -0.069 [-0.103, -0.035] (0/10)  |

## T6_nowatch: MMAct: 247 real clips without a watch signal (6 people, 3 seeds pooled; revision 1)

The watch channels are filled with the training mean. 'Acc. alone' of the first row = doing nothing. The reference row (an oracle that knows the watch is missing) is read from the scorer's log, cert_nowatch_score_d2v.log, where it is computed from the stored model outputs.

| System                                                     | Acc. alone % | Acc. after use/drop % | Flagged % | Names the watch (flagged) % | AUROC no-watch vs reference (4 people) | Acc. on reference clips % |
|------------------------------------------------------------|--------------|-----------------------|-----------|-----------------------------|----------------------------------------|---------------------------|
| Confidence (MSP)                                           | 55.9         | 55.9                  | 0.0       | --                          | 0.825                                  | 90.1                      |
| C3 (conference certificate)                                | 55.9         | 48.9                  | 85.2      | 38.2                        | 0.855                                  | 89.9                      |
| REV3 (ours, main)                                          | 55.9         | 54.0                  | 91.4      | 48.7                        | 0.945                                  | 89.1                      |
| TMC                                                        | 80.8         | 80.8                  | 8.0       | 66.1                        | 0.809                                  | 90.3                      |
| ECML                                                       | 84.1         | 83.8                  | 36.7      | 64.3                        | 0.708                                  | 89.5                      |
| Dropout-trained model                                      | 54.1         | 54.1                  | 83.9      | 46.5                        | 0.947                                  | 89.6                      |
| Backup without the watch, sensor known missing [reference] | --           | 84.1                  | --        | --                          | --                                     | --                        |

## T7_realdrop: WearGait-PD: real sensor dropouts (3,250 single-group clips x 3 seeds, pooled; revision 1)

Main result named in advance (REV3 use / drop minus doing nothing, single-group dropout clips): +0.7 points [-2.4, +3.8] over 10 folds; better in 2, level in 1, worse in 7.

| System                                                  | Acc. alone % | Acc. after use/drop % | Flagged % | False alarms (clean) % | Names right group (flagged) % | AUROC dropout vs clean (9 folds) |
|---------------------------------------------------------|--------------|-----------------------|-----------|------------------------|-------------------------------|----------------------------------|
| Confidence (MSP)                                        | 94.3         | 94.3                  | 0.0       | 0.0                    | --                            | 0.536                            |
| C3 (conference certificate)                             | 94.3         | 95.5                  | 42.4      | 25.8                   | 43.9                          | 0.546                            |
| REV3 (ours, main)                                       | 94.3         | 95.2                  | 15.6      | 6.7                    | 64.7                          | 0.591                            |
| TMC                                                     | 96.0         | 95.8                  | 8.6       | 4.4                    | 55.4                          | 0.549                            |
| ECML                                                    | 95.9         | 96.2                  | 10.6      | 6.3                    | 55.2                          | 0.546                            |
| Dropout-trained model                                   | 95.8         | 95.8                  | 7.9       | 2.5                    | 75.8                          | 0.591                            |
| Backup for the missing group, known missing [reference] | --           | 92.9                  | --        | --                     | --                            | --                               |

## T8_imuoff: WearGait-PD IMU-off test: k IMUs of one body group switched off for the whole clip (planned test; mean over 10 folds; revision 1)

Predictions P1-P5 were written before the run (docs/plan_followup_2026-11-12.md, A.6); all five were confirmed. k = all IMUs equals the simulated removal r_fill.

| Group | IMUs off | Doing nothing % | REV3 (ours, main) % | TMC % | Dropout-trained model % | REV3 minus nothing (points) | Folds better | REV3 AUROC |
|-------|----------|-----------------|---------------------|-------|-------------------------|-----------------------------|--------------|------------|
| arms  | 1        | 96.4            | 95.7                | 94.8  | 96.2                    | -0.73 [-1.25, -0.21]        | 2/10         | 0.769      |
| arms  | 2        | 96.3            | 95.6                | 92.4  | 96.2                    | -0.73 [-1.68, +0.23]        | 2/10         | 0.901      |
| trunk | 1        | 96.2            | 95.4                | 94.8  | 95.8                    | -0.76 [-1.31, -0.20]        | 1/10         | 0.653      |
| trunk | 2        | 95.7            | 95.0                | 92.2  | 95.3                    | -0.69 [-1.16, -0.22]        | 1/10         | 0.815      |
| trunk | 3        | 94.9            | 95.7                | 89.6  | 96.0                    | +0.76 [+0.17, +1.36]        | 9/10         | 0.910      |
| legs  | 1        | 95.8            | 95.6                | 96.0  | 96.1                    | -0.28 [-0.55, -0.00]        | 3/10         | 0.533      |
| legs  | 2        | 94.5            | 94.3                | 95.5  | 95.3                    | -0.14 [-0.36, +0.09]        | 4/10         | 0.596      |
| legs  | 3        | 92.6            | 92.8                | 94.7  | 94.1                    | +0.14 [-0.14, +0.42]        | 6/10         | 0.672      |
| legs  | 4        | 89.3            | 90.4                | 93.7  | 92.5                    | +1.11 [+0.49, +1.74]        | 9/10         | 0.743      |
| legs  | 5        | 83.8            | 88.4                | 92.5  | 90.7                    | +4.64 [+3.34, +5.93]        | 10/10        | 0.805      |
| legs  | 6        | 75.4            | 87.8                | 91.1  | 89.7                    | +12.37 [+10.49, +14.25]     | 10/10        | 0.855      |
| legs  | 7        | 65.5            | 89.3                | 89.2  | 89.8                    | +23.73 [+21.80, +25.67]     | 10/10        | 0.888      |
| legs  | 8        | 52.5            | 92.0                | 87.0  | 91.0                    | +39.52 [+36.47, +42.57]     | 10/10        | 0.912      |

## T9_dropout: Separate backup models vs one dropout-trained model (planted lies, revision 1)

B = hide the named sensor in the main model without retraining; C = dropout-trained model (p=0.5) as the backup; D = dropout-trained model used everywhere.

| Dataset     | Backup design             | Lie AUROC | Normal acc. alone % | Lie-clip acc. after use/drop % | Minus separate backups (points) |
|-------------|---------------------------|-----------|---------------------|--------------------------------|---------------------------------|
| UTD-MHAD    | main (separate backups)   | 0.901     | 94.7                | 75.7                           | --                              |
| UTD-MHAD    | hide, no retraining       | 0.901     | 94.7                | 69.5                           | -6.2 [-8.5, -3.9] (0/8)         |
| UTD-MHAD    | dropout as backup (sd50)  | 0.901     | 94.7                | 73.8                           | -1.8 [-4.5, +0.8] (3/8)         |
| UTD-MHAD    | dropout everywhere (sd50) | 0.901     | 94.0                | 73.4                           | -2.3 [-4.9, +0.4] (1/8)         |
| MMAct       | main (separate backups)   | 0.930     | 89.1                | 75.5                           | --                              |
| MMAct       | hide, no retraining       | 0.930     | 89.1                | 61.8                           | -13.7 [-15.4, -12.0] (0/20)     |
| MMAct       | dropout as backup (sd50)  | 0.930     | 89.1                | 74.0                           | -1.5 [-2.2, -0.8] (2/20)        |
| MMAct       | dropout everywhere (sd50) | 0.930     | 89.6                | 74.2                           | -1.3 [-2.1, -0.6] (4/20)        |
| WearGait-PD | main (separate backups)   | 0.980     | 96.6                | 93.5                           | --                              |
| WearGait-PD | hide, no retraining       | 0.980     | 96.6                | 87.5                           | -6.0 [-6.8, -5.3] (0/10)        |
| WearGait-PD | dropout as backup (sd50)  | 0.980     | 96.6                | 93.3                           | -0.2 [-0.8, +0.4] (4/10)        |
| WearGait-PD | dropout everywhere (sd50) | 0.980     | 96.9                | 93.3                           | -0.2 [-0.8, +0.3] (5/10)        |

## T10_posthoc: POST HOC rule changes (designed after seeing results; labelled extra rows only)

Rule (a): confidence as a second alarm (detection and abstention only). Rule (b): drop only when the naming margin passes a threshold chosen on validation lies. Means over units. IMU-off rows: results/certificate/posthoc_imuoff_per_fold.csv.

| Bench    | Dataset     | Variant                                | measure 1                  | measure 2               |
|----------|-------------|----------------------------------------|----------------------------|-------------------------|
| lies     | UTD-MHAD    | REV3 (main)                            | Lie AUROC: 0.901           | Lie-clip acc. %: 75.7   |
| lies     | UTD-MHAD    | REV3 + (a) confidence alarm [post hoc] | Lie AUROC: 0.875           | Lie-clip acc. %: 75.7   |
| lies     | UTD-MHAD    | REV3 + (b) sure-naming gate [post hoc] | Lie AUROC: 0.901           | Lie-clip acc. %: 74.7   |
| lies     | UTD-MHAD    | REV3 + (a) + (b) [post hoc]            | Lie AUROC: 0.875           | Lie-clip acc. %: 74.7   |
| lies     | UTD-MHAD    | confidence (1 - msp) [reference]       | Lie AUROC: 0.726           | Lie-clip acc. %: 60.8   |
| faults   | UTD-MHAD    | REV3 (main)                            | Harmful-fault AUROC: 0.838 | Fault-clip acc. %: 87.5 |
| faults   | UTD-MHAD    | REV3 + (a) confidence alarm [post hoc] | Harmful-fault AUROC: 0.900 | Fault-clip acc. %: 87.5 |
| faults   | UTD-MHAD    | REV3 + (b) sure-naming gate [post hoc] | Harmful-fault AUROC: 0.838 | Fault-clip acc. %: 88.6 |
| faults   | UTD-MHAD    | REV3 + (a) + (b) [post hoc]            | Harmful-fault AUROC: 0.900 | Fault-clip acc. %: 88.6 |
| faults   | UTD-MHAD    | confidence (1 - msp) [reference]       | Harmful-fault AUROC: 0.926 | Fault-clip acc. %: 90.7 |
| lies     | MMAct       | REV3 (main)                            | Lie AUROC: 0.930           | Lie-clip acc. %: 75.5   |
| lies     | MMAct       | REV3 + (a) confidence alarm [post hoc] | Lie AUROC: 0.910           | Lie-clip acc. %: 75.5   |
| lies     | MMAct       | REV3 + (b) sure-naming gate [post hoc] | Lie AUROC: 0.930           | Lie-clip acc. %: 75.3   |
| lies     | MMAct       | REV3 + (a) + (b) [post hoc]            | Lie AUROC: 0.910           | Lie-clip acc. %: 75.3   |
| lies     | MMAct       | confidence (1 - msp) [reference]       | Lie AUROC: 0.675           | Lie-clip acc. %: 54.2   |
| faults   | MMAct       | REV3 (main)                            | Harmful-fault AUROC: 0.774 | Fault-clip acc. %: 86.0 |
| faults   | MMAct       | REV3 + (a) confidence alarm [post hoc] | Harmful-fault AUROC: 0.873 | Fault-clip acc. %: 86.0 |
| faults   | MMAct       | REV3 + (b) sure-naming gate [post hoc] | Harmful-fault AUROC: 0.774 | Fault-clip acc. %: 86.2 |
| faults   | MMAct       | REV3 + (a) + (b) [post hoc]            | Harmful-fault AUROC: 0.873 | Fault-clip acc. %: 86.2 |
| faults   | MMAct       | confidence (1 - msp) [reference]       | Harmful-fault AUROC: 0.898 | Fault-clip acc. %: 87.6 |
| lies     | WearGait-PD | REV3 (main)                            | Lie AUROC: 0.980           | Lie-clip acc. %: 93.5   |
| lies     | WearGait-PD | REV3 + (a) confidence alarm [post hoc] | Lie AUROC: 0.977           | Lie-clip acc. %: 93.5   |
| lies     | WearGait-PD | REV3 + (b) sure-naming gate [post hoc] | Lie AUROC: 0.980           | Lie-clip acc. %: 93.5   |
| lies     | WearGait-PD | REV3 + (a) + (b) [post hoc]            | Lie AUROC: 0.977           | Lie-clip acc. %: 93.5   |
| lies     | WearGait-PD | confidence (1 - msp) [reference]       | Lie AUROC: 0.618           | Lie-clip acc. %: 65.9   |
| faults   | WearGait-PD | REV3 (main)                            | Harmful-fault AUROC: 0.862 | Fault-clip acc. %: 95.9 |
| faults   | WearGait-PD | REV3 + (a) confidence alarm [post hoc] | Harmful-fault AUROC: 0.950 | Fault-clip acc. %: 95.9 |
| faults   | WearGait-PD | REV3 + (b) sure-naming gate [post hoc] | Harmful-fault AUROC: 0.862 | Fault-clip acc. %: 96.0 |
| faults   | WearGait-PD | REV3 + (a) + (b) [post hoc]            | Harmful-fault AUROC: 0.950 | Fault-clip acc. %: 96.0 |
| faults   | WearGait-PD | confidence (1 - msp) [reference]       | Harmful-fault AUROC: 0.947 | Fault-clip acc. %: 96.2 |
| nowatch  | MMAct       | REV3 (main)                            | No-watch AUROC: 0.945      | No-watch acc. %: 54.0   |
| nowatch  | MMAct       | REV3 + (a) confidence alarm [post hoc] | No-watch AUROC: 0.947      | No-watch acc. %: 54.0   |
| nowatch  | MMAct       | REV3 + (b) sure-naming gate [post hoc] | No-watch AUROC: 0.945      | No-watch acc. %: 54.1   |
| nowatch  | MMAct       | REV3 + (a) + (b) [post hoc]            | No-watch AUROC: 0.947      | No-watch acc. %: 54.1   |
| nowatch  | MMAct       | confidence (1 - msp) [reference]       | No-watch AUROC: 0.825      | No-watch acc. %: 55.9   |
| realdrop | WearGait-PD | REV3 (main)                            | Dropout AUROC: 0.591       | Dropout acc. %: 95.6    |
| realdrop | WearGait-PD | REV3 + (a) confidence alarm [post hoc] | Dropout AUROC: 0.577       | Dropout acc. %: 95.6    |
| realdrop | WearGait-PD | REV3 + (b) sure-naming gate [post hoc] | Dropout AUROC: 0.591       | Dropout acc. %: 95.8    |
| realdrop | WearGait-PD | REV3 + (a) + (b) [post hoc]            | Dropout AUROC: 0.577       | Dropout acc. %: 95.8    |
| realdrop | WearGait-PD | confidence (1 - msp) [reference]       | Dropout AUROC: 0.536       | Dropout acc. %: 94.9    |

## T11_v1_signs: Ablation: the conference warning signs (v1) next to the journal signals (copied from the scorer's summary files)

| Dataset     | task  | score                     | auroc                       | minus_msp                      | subjects_better |
|-------------|-------|---------------------------|-----------------------------|--------------------------------|-----------------|
| UTD-MHAD    | LIE   | 1 - msp (baseline)        | 0.726 [0.705, 0.748] (n=8)  | --                             | --              |
| UTD-MHAD    | LIE   | COMBINED (S1+S2+msp)      | 0.781 [0.751, 0.810] (n=8)  | 0.054 [0.040, 0.068] (n=8)     | 8/8             |
| UTD-MHAD    | LIE   | COMBINED v2 (experts+msp) | 0.871 [0.837, 0.906] (n=8)  | 0.145 [0.123, 0.167] (n=8)     | 8/8             |
| UTD-MHAD    | LIE   | S1 stability drop         | 0.574 [0.560, 0.587] (n=8)  | -0.153 [-0.172, -0.134] (n=8)  | 0/8             |
| UTD-MHAD    | LIE   | S1 stability flip         | 0.562 [0.552, 0.571] (n=8)  | -0.165 [-0.187, -0.143] (n=8)  | 0/8             |
| UTD-MHAD    | LIE   | S2 hide-one flips         | 0.717 [0.684, 0.750] (n=8)  | -0.009 [-0.030, 0.011] (n=8)   | 2/8             |
| UTD-MHAD    | LIE   | S2 hide-one shift         | 0.663 [0.621, 0.704] (n=8)  | -0.064 [-0.093, -0.034] (n=8)  | 0/8             |
| UTD-MHAD    | LIE   | X experts disagree (JS)   | 0.855 [0.820, 0.890] (n=8)  | 0.129 [0.100, 0.157] (n=8)     | 8/8             |
| UTD-MHAD    | LIE   | X lowest expert support   | 0.899 [0.865, 0.932] (n=8)  | 0.172 [0.154, 0.191] (n=8)     | 8/8             |
| UTD-MHAD    | LIE   | X mean expert support     | 0.786 [0.758, 0.815] (n=8)  | 0.060 [0.037, 0.083] (n=8)     | 8/8             |
| UTD-MHAD    | LIE   | energy (baseline)         | 0.714 [0.688, 0.741] (n=8)  | -0.012 [-0.025, 0.001] (n=8)   | 2/8             |
| UTD-MHAD    | LIE   | entropy (baseline)        | 0.726 [0.701, 0.750] (n=8)  | -0.001 [-0.007, 0.005] (n=8)   | 4/8             |
| UTD-MHAD    | ERROR | 1 - msp (baseline)        | 0.926 [0.913, 0.938] (n=7)  | --                             | --              |
| UTD-MHAD    | ERROR | COMBINED (S1+S2+msp)      | 0.900 [0.864, 0.935] (n=6)  | -0.023 [-0.056, 0.011] (n=6)   | 1/6             |
| UTD-MHAD    | ERROR | COMBINED v2 (experts+msp) | 0.912 [0.872, 0.951] (n=6)  | -0.011 [-0.042, 0.019] (n=6)   | 3/6             |
| UTD-MHAD    | ERROR | S1 stability drop         | 0.625 [0.532, 0.718] (n=7)  | -0.300 [-0.404, -0.196] (n=7)  | 0/7             |
| UTD-MHAD    | ERROR | S1 stability flip         | 0.653 [0.597, 0.708] (n=7)  | -0.273 [-0.338, -0.207] (n=7)  | 0/7             |
| UTD-MHAD    | ERROR | S2 hide-one flips         | 0.786 [0.744, 0.829] (n=7)  | -0.139 [-0.175, -0.104] (n=7)  | 0/7             |
| UTD-MHAD    | ERROR | S2 hide-one shift         | 0.498 [0.454, 0.542] (n=7)  | -0.428 [-0.469, -0.386] (n=7)  | 0/7             |
| UTD-MHAD    | ERROR | X experts disagree (JS)   | 0.668 [0.585, 0.752] (n=7)  | -0.257 [-0.341, -0.173] (n=7)  | 0/7             |
| UTD-MHAD    | ERROR | X lowest expert support   | 0.853 [0.792, 0.914] (n=7)  | -0.073 [-0.135, -0.010] (n=7)  | 1/7             |
| UTD-MHAD    | ERROR | X mean expert support     | 0.959 [0.939, 0.979] (n=7)  | 0.033 [0.008, 0.058] (n=7)     | 6/7             |
| UTD-MHAD    | ERROR | energy (baseline)         | 0.877 [0.857, 0.898] (n=7)  | -0.048 [-0.071, -0.025] (n=7)  | 0/7             |
| UTD-MHAD    | ERROR | entropy (baseline)        | 0.908 [0.892, 0.925] (n=7)  | -0.017 [-0.032, -0.003] (n=7)  | 0/7             |
| MMAct       | LIE   | 1 - msp (baseline)        | 0.675 [0.662, 0.687] (n=20) | --                             | --              |
| MMAct       | LIE   | COMBINED (S1+S2+msp)      | 0.722 [0.707, 0.737] (n=20) | 0.047 [0.039, 0.055] (n=20)    | 20/20           |
| MMAct       | LIE   | COMBINED v2 (experts+msp) | 0.850 [0.828, 0.871] (n=20) | 0.175 [0.161, 0.190] (n=20)    | 20/20           |
| MMAct       | LIE   | S1 stability drop         | 0.599 [0.589, 0.609] (n=20) | -0.075 [-0.082, -0.069] (n=20) | 0/20            |
| MMAct       | LIE   | S1 stability flip         | 0.555 [0.549, 0.560] (n=20) | -0.120 [-0.130, -0.111] (n=20) | 0/20            |
| MMAct       | LIE   | S2 hide-one flips         | 0.640 [0.628, 0.652] (n=20) | -0.035 [-0.043, -0.026] (n=20) | 0/20            |
| MMAct       | LIE   | S2 hide-one shift         | 0.630 [0.613, 0.647] (n=20) | -0.045 [-0.059, -0.031] (n=20) | 0/20            |
| MMAct       | LIE   | X experts disagree (JS)   | 0.846 [0.825, 0.867] (n=20) | 0.171 [0.157, 0.186] (n=20)    | 20/20           |
| MMAct       | LIE   | X lowest expert support   | 0.879 [0.860, 0.898] (n=20) | 0.204 [0.191, 0.217] (n=20)    | 20/20           |
| MMAct       | LIE   | X mean expert support     | 0.804 [0.786, 0.823] (n=20) | 0.130 [0.118, 0.141] (n=20)    | 20/20           |
| MMAct       | LIE   | energy (baseline)         | 0.666 [0.655, 0.678] (n=20) | -0.008 [-0.011, -0.006] (n=20) | 2/20            |
| MMAct       | LIE   | entropy (baseline)        | 0.677 [0.664, 0.689] (n=20) | 0.002 [0.001, 0.002] (n=20)    | 20/20           |
| MMAct       | ERROR | 1 - msp (baseline)        | 0.859 [0.838, 0.880] (n=20) | --                             | --              |
| MMAct       | ERROR | COMBINED (S1+S2+msp)      | 0.831 [0.808, 0.853] (n=20) | -0.028 [-0.045, -0.012] (n=20) | 4/20            |
| MMAct       | ERROR | COMBINED v2 (experts+msp) | 0.846 [0.815, 0.876] (n=20) | -0.013 [-0.036, 0.010] (n=20)  | 7/20            |
| MMAct       | ERROR | S1 stability drop         | 0.746 [0.721, 0.771] (n=20) | -0.112 [-0.146, -0.079] (n=20) | 1/20            |
| MMAct       | ERROR | S1 stability flip         | 0.696 [0.672, 0.720] (n=20) | -0.163 [-0.183, -0.142] (n=20) | 0/20            |
| MMAct       | ERROR | S2 hide-one flips         | 0.699 [0.674, 0.723] (n=20) | -0.160 [-0.178, -0.143] (n=20) | 0/20            |
| MMAct       | ERROR | S2 hide-one shift         | 0.526 [0.507, 0.546] (n=20) | -0.332 [-0.358, -0.307] (n=20) | 0/20            |
| MMAct       | ERROR | X experts disagree (JS)   | 0.698 [0.670, 0.727] (n=20) | -0.160 [-0.190, -0.130] (n=20) | 0/20            |
| MMAct       | ERROR | X lowest expert support   | 0.833 [0.803, 0.862] (n=20) | -0.026 [-0.049, -0.003] (n=20) | 6/20            |
| MMAct       | ERROR | X mean expert support     | 0.896 [0.869, 0.922] (n=20) | 0.037 [0.016, 0.058] (n=20)    | 17/20           |
| MMAct       | ERROR | energy (baseline)         | 0.835 [0.810, 0.859] (n=20) | -0.024 [-0.037, -0.012] (n=20) | 4/20            |
| MMAct       | ERROR | entropy (baseline)        | 0.858 [0.837, 0.879] (n=20) | -0.001 [-0.002, 0.000] (n=20)  | 7/20            |
| WearGait-PD | LIE   | 1 - msp (baseline)        | 0.618 [0.604, 0.632] (n=10) | --                             | --              |
| WearGait-PD | LIE   | COMBINED (S1+S2+msp)      | 0.772 [0.762, 0.782] (n=10) | 0.154 [0.139, 0.169] (n=10)    | 10/10           |
| WearGait-PD | LIE   | COMBINED v2 (experts+msp) | 0.965 [0.955, 0.975] (n=10) | 0.347 [0.334, 0.360] (n=10)    | 10/10           |
| WearGait-PD | LIE   | S1 stability drop         | 0.621 [0.609, 0.633] (n=10) | 0.003 [-0.005, 0.011] (n=10)   | 5/10            |
| WearGait-PD | LIE   | S1 stability flip         | 0.551 [0.544, 0.558] (n=10) | -0.067 [-0.079, -0.056] (n=10) | 0/10            |
| WearGait-PD | LIE   | S2 hide-one flips         | 0.715 [0.707, 0.722] (n=10) | 0.096 [0.082, 0.111] (n=10)    | 10/10           |
| WearGait-PD | LIE   | S2 hide-one shift         | 0.764 [0.754, 0.774] (n=10) | 0.146 [0.128, 0.164] (n=10)    | 10/10           |
| WearGait-PD | LIE   | X experts disagree (JS)   | 0.969 [0.959, 0.978] (n=10) | 0.350 [0.337, 0.364] (n=10)    | 10/10           |
| WearGait-PD | LIE   | X lowest expert support   | 0.970 [0.961, 0.979] (n=10) | 0.352 [0.339, 0.364] (n=10)    | 10/10           |
| WearGait-PD | LIE   | X mean expert support     | 0.930 [0.914, 0.946] (n=10) | 0.312 [0.296, 0.328] (n=10)    | 10/10           |
| WearGait-PD | LIE   | energy (baseline)         | 0.577 [0.566, 0.588] (n=10) | -0.041 [-0.051, -0.031] (n=10) | 0/10            |
| WearGait-PD | LIE   | entropy (baseline)        | 0.618 [0.604, 0.631] (n=10) | -0.001 [-0.001, -0.000] (n=10) | 2/10            |
| WearGait-PD | ERROR | 1 - msp (baseline)        | 0.902 [0.879, 0.925] (n=10) | --                             | --              |
| WearGait-PD | ERROR | COMBINED (S1+S2+msp)      | 0.881 [0.854, 0.907] (n=10) | -0.021 [-0.048, 0.005] (n=10)  | 3/10            |
| WearGait-PD | ERROR | COMBINED v2 (experts+msp) | 0.931 [0.914, 0.948] (n=10) | 0.029 [0.005, 0.053] (n=10)    | 8/10            |
| WearGait-PD | ERROR | S1 stability drop         | 0.853 [0.825, 0.881] (n=10) | -0.049 [-0.084, -0.014] (n=10) | 2/10            |
| WearGait-PD | ERROR | S1 stability flip         | 0.766 [0.736, 0.796] (n=10) | -0.136 [-0.165, -0.106] (n=10) | 0/10            |
| WearGait-PD | ERROR | S2 hide-one flips         | 0.718 [0.684, 0.751] (n=10) | -0.184 [-0.218, -0.150] (n=10) | 0/10            |
| WearGait-PD | ERROR | S2 hide-one shift         | 0.737 [0.712, 0.762] (n=10) | -0.165 [-0.191, -0.139] (n=10) | 0/10            |
| WearGait-PD | ERROR | X experts disagree (JS)   | 0.863 [0.848, 0.879] (n=10) | -0.038 [-0.066, -0.011] (n=10) | 2/10            |
| WearGait-PD | ERROR | X lowest expert support   | 0.915 [0.892, 0.938] (n=10) | 0.013 [-0.020, 0.046] (n=10)   | 5/10            |
| WearGait-PD | ERROR | X mean expert support     | 0.932 [0.912, 0.952] (n=10) | 0.030 [0.000, 0.060] (n=10)    | 6/10            |
| WearGait-PD | ERROR | energy (baseline)         | 0.884 [0.853, 0.914] (n=10) | -0.018 [-0.030, -0.007] (n=10) | 1/10            |
| WearGait-PD | ERROR | entropy (baseline)        | 0.901 [0.877, 0.925] (n=10) | -0.001 [-0.001, 0.000] (n=10)  | 2/10            |

## T12_mi_bound: Dataset-level mutual information between sensors (InfoNCE lower bound, own matcher)

| Dataset     | Sensor pair | MI lower bound (nats) | Folds | Range        |
|-------------|-------------|-----------------------|-------|--------------|
| UTD-MHAD    | skel-imu    | 1.30 [1.09, 1.52]     | 8     | 0.92 to 1.79 |
| UTD-MHAD    | video-imu   | 0.89 [0.55, 1.23]     | 8     | 0.45 to 1.51 |
| UTD-MHAD    | video-skel  | 0.87 [0.50, 1.25]     | 8     | 0.21 to 1.64 |
| MMAct       | phone-video | 1.30 [1.10, 1.50]     | 20    | 0.41 to 1.77 |
| MMAct       | watch-phone | 1.33 [1.16, 1.50]     | 20    | 0.41 to 1.74 |
| MMAct       | watch-video | 1.61 [1.43, 1.80]     | 20    | 0.21 to 2.05 |
| WearGait-PD | arms-legs   | 1.72 [1.65, 1.79]     | 10    | 1.54 to 1.83 |
| WearGait-PD | arms-trunk  | 1.76 [1.70, 1.82]     | 10    | 1.63 to 1.86 |
| WearGait-PD | trunk-legs  | 2.06 [2.02, 2.10]     | 10    | 1.95 to 2.14 |

## T13_faults_new: Two more everyday faults, stuck sensor and saturation (frozen transfer, the new faults pooled, revision 1)

Stuck = the sensor freezes at 75 / 50 / 0 % of the clip until its end; saturation = readings clipped at +/-8 / 4 / 2 g and +/-1000 / 500 / 250 deg/s; video: frozen frames and over-exposure x1.5 / 2 / 3. Certificate fitted on validation lies only (frozen). The original faults (T5) are not part of this pool.

| Dataset     | System                      | AUROC, harmful faults | AUROC, all faults | Names faulty sensor % (harmful) | Fault-clip acc. %: alone -> use/drop | False alarms on clean % |
|-------------|-----------------------------|-----------------------|-------------------|---------------------------------|--------------------------------------|-------------------------|
| UTD-MHAD    | Confidence (MSP)            | 0.922 [0.896, 0.948]  | 0.578             | --                              | 88.3 -> 88.3                         | 0.0                     |
| UTD-MHAD    | C3 (conference certificate) | 0.918 [0.899, 0.938]  | 0.620             | 70.1                            | 88.3 -> 85.7                         | 20.7                    |
| UTD-MHAD    | REV3 (ours, main)           | 0.900 [0.864, 0.935]  | 0.642             | 70.5                            | 88.3 -> 86.8                         | 25.0                    |
| UTD-MHAD    | TMC                         | 0.881 [0.854, 0.909]  | 0.593             | 17.0                            | 90.9 -> 90.9                         | 9.5                     |
| UTD-MHAD    | ECML                        | 0.872 [0.842, 0.902]  | 0.593             | 60.9                            | 87.0 -> 87.4                         | 12.1                    |
| UTD-MHAD    | Dropout-trained model       | 0.905 [0.868, 0.941]  | 0.641             | 66.9                            | 89.8 -> 88.0                         | 12.5                    |
| MMAct       | Confidence (MSP)            | 0.883 [0.870, 0.897]  | 0.584             | --                              | 79.0 -> 79.0                         | 0.0                     |
| MMAct       | C3 (conference certificate) | 0.861 [0.844, 0.878]  | 0.608             | 64.3                            | 79.0 -> 81.8                         | 43.8                    |
| MMAct       | REV3 (ours, main)           | 0.876 [0.856, 0.896]  | 0.632             | 66.3                            | 79.0 -> 81.2                         | 30.5                    |
| MMAct       | TMC                         | 0.848 [0.825, 0.871]  | 0.577             | 11.4                            | 82.4 -> 82.4                         | 1.4                     |
| MMAct       | ECML                        | 0.829 [0.811, 0.847]  | 0.573             | 8.2                             | 79.9 -> 79.9                         | 5.1                     |
| MMAct       | Dropout-trained model       | 0.876 [0.856, 0.896]  | 0.631             | 64.4                            | 81.6 -> 82.3                         | 13.2                    |
| WearGait-PD | Confidence (MSP)            | 0.909 [0.887, 0.931]  | 0.519             | --                              | 95.2 -> 95.2                         | 0.0                     |
| WearGait-PD | C3 (conference certificate) | 0.837 [0.824, 0.851]  | 0.558             | 68.6                            | 95.2 -> 95.4                         | 26.4                    |
| WearGait-PD | REV3 (ours, main)           | 0.941 [0.931, 0.951]  | 0.606             | 82.0                            | 95.2 -> 95.6                         | 6.8                     |
| WearGait-PD | TMC                         | 0.930 [0.918, 0.941]  | 0.547             | 56.1                            | 95.9 -> 95.6                         | 4.4                     |
| WearGait-PD | ECML                        | 0.881 [0.831, 0.932]  | 0.545             | 60.8                            | 95.5 -> 95.2                         | 6.3                     |
| WearGait-PD | Dropout-trained model       | 0.933 [0.922, 0.943]  | 0.607             | 76.6                            | 96.2 -> 96.2                         | 2.6                     |

## T13b_faults_new_paired: Stuck sensor and saturation: REV3 minus other systems, harmful-fault AUROC, paired per unit

| Dataset     | REV3 minus                  | AUROC, harmful faults           |
|-------------|-----------------------------|---------------------------------|
| UTD-MHAD    | Confidence (MSP)            | -0.022 [-0.045, +0.001] (2/8)   |
| UTD-MHAD    | C3 (conference certificate) | -0.018 [-0.053, +0.016] (4/8)   |
| UTD-MHAD    | TMC                         | +0.019 [-0.005, +0.043] (6/8)   |
| MMAct       | Confidence (MSP)            | -0.007 [-0.023, +0.009] (9/20)  |
| MMAct       | C3 (conference certificate) | +0.015 [+0.003, +0.026] (15/20) |
| MMAct       | TMC                         | +0.028 [+0.012, +0.044] (18/20) |
| WearGait-PD | Confidence (MSP)            | +0.032 [+0.007, +0.057] (9/10)  |
| WearGait-PD | C3 (conference certificate) | +0.103 [+0.093, +0.113] (10/10) |
| WearGait-PD | TMC                         | +0.011 [-0.002, +0.024] (8/10)  |

## T13c_faults_new_by_level: Stuck sensor and saturation by fault and level, all sensors pooled (means over units, revision 1)

| Dataset     | Fault      | Level | Answers changed % | Fault-clip acc. alone % | REV3 use/drop % | AUROC harmful: confidence | AUROC harmful: REV3 | REV3 names faulty sensor % (harmful) |
|-------------|------------|-------|-------------------|-------------------------|-----------------|---------------------------|---------------------|--------------------------------------|
| UTD-MHAD    | saturation | 1     | 1.5               | 94.4                    | 90.0            | 0.955                     | 0.836               | 68.5                                 |
| UTD-MHAD    | saturation | 2     | 2.1               | 94.2                    | 89.5            | 0.941                     | 0.862               | 56.6                                 |
| UTD-MHAD    | saturation | 3     | 3.1               | 93.7                    | 89.3            | 0.942                     | 0.880               | 65.2                                 |
| UTD-MHAD    | stuck      | 1     | 1.4               | 94.5                    | 91.0            | 0.960                     | 0.768               | 34.9                                 |
| UTD-MHAD    | stuck      | 2     | 6.9               | 90.4                    | 87.2            | 0.903                     | 0.822               | 47.4                                 |
| UTD-MHAD    | stuck      | 3     | 29.5              | 68.6                    | 76.4            | 0.921                     | 0.929               | 78.8                                 |
| MMAct       | saturation | 1     | 5.1               | 85.9                    | 84.7            | 0.911                     | 0.795               | 51.2                                 |
| MMAct       | saturation | 2     | 12.4              | 79.5                    | 80.8            | 0.915                     | 0.845               | 60.3                                 |
| MMAct       | saturation | 3     | 18.5              | 74.1                    | 78.4            | 0.906                     | 0.870               | 65.5                                 |
| MMAct       | stuck      | 1     | 3.3               | 88.1                    | 86.3            | 0.904                     | 0.730               | 38.7                                 |
| MMAct       | stuck      | 2     | 10.4              | 82.8                    | 82.2            | 0.858                     | 0.804               | 55.0                                 |
| MMAct       | stuck      | 3     | 32.4              | 63.5                    | 75.1            | 0.860                     | 0.940               | 77.7                                 |
| WearGait-PD | saturation | 1     | 0.0               | 96.6                    | 96.4            | --                        | --                  | --                                   |
| WearGait-PD | saturation | 2     | 0.0               | 96.6                    | 96.4            | --                        | --                  | --                                   |
| WearGait-PD | saturation | 3     | 0.1               | 96.6                    | 96.4            | 0.993                     | 0.846               | 24.6                                 |
| WearGait-PD | stuck      | 1     | 0.9               | 96.3                    | 96.0            | 0.953                     | 0.831               | 35.9                                 |
| WearGait-PD | stuck      | 2     | 2.0               | 95.6                    | 95.2            | 0.933                     | 0.874               | 58.7                                 |
| WearGait-PD | stuck      | 3     | 8.4               | 89.6                    | 93.4            | 0.898                     | 0.968               | 92.6                                 |

## T13d_saturation_clipped: Saturation: share of readings actually clipped (test part, all folds)

Levels: 1 = +/-8 g, +/-1000 deg/s; 2 = +/-4 g, +/-500 deg/s; 3 = +/-2 g, +/-250 deg/s (scripts/faults2_clip_share.py).

| Dataset     | Sensor | Level | Readings clipped % | Clips with any clipping % |
|-------------|--------|-------|--------------------|---------------------------|
| UTD-MHAD    | IMU    | 1     | 0.004              | 1.5                       |
| UTD-MHAD    | IMU    | 2     | 0.257              | 30.1                      |
| UTD-MHAD    | IMU    | 3     | 3.091              | 76.7                      |
| MMAct       | watch  | 1     | 0.002              | 1.5                       |
| MMAct       | watch  | 2     | 0.101              | 11.8                      |
| MMAct       | watch  | 3     | 0.706              | 33.9                      |
| MMAct       | phone  | 1     | 0.000              | 0.0                       |
| MMAct       | phone  | 2     | 0.014              | 8.0                       |
| MMAct       | phone  | 3     | 0.324              | 44.2                      |
| WearGait-PD | arms   | 1     | 0.000              | 0.1                       |
| WearGait-PD | arms   | 2     | 0.009              | 2.1                       |
| WearGait-PD | arms   | 3     | 0.235              | 16.4                      |
| WearGait-PD | trunk  | 1     | 0.000              | 0.0                       |
| WearGait-PD | trunk  | 2     | 0.000              | 0.1                       |
| WearGait-PD | trunk  | 3     | 0.007              | 4.2                       |
| WearGait-PD | legs   | 1     | 0.000              | 0.3                       |
| WearGait-PD | legs   | 2     | 0.034              | 23.0                      |
| WearGait-PD | legs   | 3     | 1.298              | 59.2                      |

## T14_nowatch_rawzero: MMAct: the 247 real no-watch clips with the watch as raw 0 (addition 3; 6 people, 3 seeds pooled; revision 1)

Raw 0 on every watch channel, then the training normalisation (physically impossible while worn). REV3 per person: person 2: 31.5% vs doing nothing 70.6% (126 clips); person 10: 86.7% vs doing nothing 40.0% (5 clips); person 11: 83.3% vs doing nothing 72.2% (6 clips); person 13: 78.6% vs doing nothing 61.9% (28 clips); person 16: 81.5% vs doing nothing 73.5% (54 clips); person 20: 61.9% vs doing nothing 61.9% (28 clips). The reference row is read from score_nowatch2_d2v.log.

| System                                                     | Acc. alone % | Acc. after use/drop % | Flagged % | Names the watch (flagged) % | AUROC no-watch vs reference (4 people) | Acc. on reference clips % |
|------------------------------------------------------------|--------------|-----------------------|-----------|-----------------------------|----------------------------------------|---------------------------|
| Confidence (MSP)                                           | 68.7         | 68.7                  | 0.0       | --                          | 0.757                                  | 90.1                      |
| C3 (conference certificate)                                | 68.7         | 50.7                  | 87.2      | 36.1                        | 0.848                                  | 89.9                      |
| REV3 (ours, main)                                          | 68.7         | 53.6                  | 90.4      | 45.7                        | 0.949                                  | 89.1                      |
| TMC                                                        | 85.0         | 85.0                  | 2.4       | 44.4                        | 0.613                                  | 90.3                      |
| ECML                                                       | 82.3         | 81.0                  | 21.3      | 49.4                        | 0.572                                  | 89.5                      |
| Dropout-trained model                                      | 64.9         | 51.3                  | 87.2      | 44.0                        | 0.951                                  | 89.6                      |
| Backup without the watch, sensor known missing [reference] | --           | 84.1                  | --        | --                          | --                                     | --                        |

## T15_imuoff_partial: WearGait-PD: k IMUs switched off during part of the clip (addition 2; mean over 10 folds; revision 1)

p = 100 % (whole clip) is table T8.

| Group | IMUs off | Off for % of the clip | Doing nothing % | REV3 (ours, main) % | REV3 minus nothing (points) | Folds better | REV3 AUROC |
|-------|----------|-----------------------|-----------------|---------------------|-----------------------------|--------------|------------|
| arms  | 1        | 25                    | 96.6            | 96.3                | -0.24 [-0.53, +0.06]        | 1/10         | 0.570      |
| arms  | 2        | 25                    | 96.6            | 96.3                | -0.28 [-0.58, +0.01]        | 2/10         | 0.698      |
| trunk | 1        | 25                    | 96.6            | 96.3                | -0.23 [-0.55, +0.08]        | 1/10         | 0.530      |
| trunk | 2        | 25                    | 96.5            | 96.2                | -0.32 [-0.69, +0.05]        | 1/10         | 0.584      |
| trunk | 3        | 25                    | 96.5            | 96.1                | -0.41 [-0.83, +0.01]        | 1/10         | 0.651      |
| legs  | 1        | 25                    | 96.5            | 96.3                | -0.17 [-0.39, +0.06]        | 1/10         | 0.504      |
| legs  | 2        | 25                    | 96.4            | 96.2                | -0.16 [-0.43, +0.10]        | 2/10         | 0.511      |
| legs  | 3        | 25                    | 96.2            | 96.0                | -0.20 [-0.44, +0.04]        | 1/10         | 0.524      |
| legs  | 4        | 25                    | 96.1            | 95.9                | -0.20 [-0.42, +0.02]        | 1/10         | 0.545      |
| legs  | 5        | 25                    | 96.0            | 95.8                | -0.25 [-0.50, -0.01]        | 1/10         | 0.570      |
| legs  | 6        | 25                    | 95.8            | 95.6                | -0.24 [-0.47, -0.01]        | 2/10         | 0.598      |
| legs  | 7        | 25                    | 95.7            | 95.5                | -0.23 [-0.43, -0.03]        | 1/10         | 0.625      |
| legs  | 8        | 25                    | 95.6            | 95.4                | -0.21 [-0.44, +0.02]        | 2/10         | 0.647      |
| arms  | 1        | 50                    | 96.6            | 96.3                | -0.28 [-0.60, +0.03]        | 2/10         | 0.641      |
| arms  | 2        | 50                    | 96.5            | 96.1                | -0.38 [-0.86, +0.11]        | 2/10         | 0.790      |
| trunk | 1        | 50                    | 96.5            | 96.1                | -0.38 [-0.78, +0.01]        | 1/10         | 0.565      |
| trunk | 2        | 50                    | 96.4            | 95.7                | -0.68 [-1.25, -0.11]        | 1/10         | 0.657      |
| trunk | 3        | 50                    | 96.2            | 95.4                | -0.79 [-1.45, -0.14]        | 1/10         | 0.738      |
| legs  | 1        | 50                    | 96.4            | 96.2                | -0.21 [-0.45, +0.04]        | 1/10         | 0.509      |
| legs  | 2        | 50                    | 96.1            | 95.8                | -0.27 [-0.55, -0.00]        | 2/10         | 0.529      |
| legs  | 3        | 50                    | 95.7            | 95.4                | -0.29 [-0.53, -0.05]        | 2/10         | 0.563      |
| legs  | 4        | 50                    | 95.2            | 95.0                | -0.24 [-0.43, -0.05]        | 2/10         | 0.605      |
| legs  | 5        | 50                    | 94.6            | 94.4                | -0.18 [-0.36, -0.01]        | 1/10         | 0.646      |
| legs  | 6        | 50                    | 93.7            | 93.7                | -0.02 [-0.37, +0.33]        | 3/10         | 0.682      |
| legs  | 7        | 50                    | 92.9            | 93.2                | +0.29 [-0.25, +0.83]        | 6/10         | 0.711      |
| legs  | 8        | 50                    | 92.1            | 93.1                | +1.06 [+0.08, +2.05]        | 8/10         | 0.730      |
| arms  | 1        | 75                    | 96.5            | 96.1                | -0.42 [-0.82, -0.01]        | 2/10         | 0.714      |
| arms  | 2        | 75                    | 96.4            | 95.9                | -0.54 [-1.29, +0.21]        | 2/10         | 0.859      |
| trunk | 1        | 75                    | 96.4            | 95.8                | -0.64 [-1.15, -0.12]        | 1/10         | 0.605      |
| trunk | 2        | 75                    | 96.2            | 95.1                | -1.08 [-1.75, -0.41]        | 1/10         | 0.734      |
| trunk | 3        | 75                    | 95.9            | 95.2                | -0.70 [-1.39, -0.01]        | 2/10         | 0.828      |
| legs  | 1        | 75                    | 96.2            | 96.0                | -0.22 [-0.46, +0.02]        | 2/10         | 0.518      |
| legs  | 2        | 75                    | 95.6            | 95.3                | -0.26 [-0.50, -0.02]        | 1/10         | 0.559      |
| legs  | 3        | 75                    | 94.8            | 94.5                | -0.24 [-0.44, -0.03]        | 2/10         | 0.616      |
| legs  | 4        | 75                    | 93.4            | 93.3                | -0.14 [-0.40, +0.13]        | 3/10         | 0.673      |
| legs  | 5        | 75                    | 91.5            | 91.8                | +0.38 [-0.40, +1.16]        | 6/10         | 0.724      |
| legs  | 6        | 75                    | 88.8            | 90.4                | +1.66 [+0.32, +3.00]        | 10/10        | 0.767      |
| legs  | 7        | 75                    | 85.7            | 89.6                | +3.90 [+2.11, +5.69]        | 10/10        | 0.801      |
| legs  | 8        | 75                    | 81.9            | 89.8                | +7.85 [+4.91, +10.80]       | 10/10        | 0.824      |

## T16_predictions_additions: Predictions written before the runs (13-14 Nov) and their verdicts

Plans: docs/plan_v1_d3_2026-11-13.md and docs/plan_additions_2026-11-14.md. A prediction that failed is reported as failed; nothing was re-tuned after a result.

| Experiment                      | Prediction | Verdict       | Result                                                                                                                                                      |
|---------------------------------|------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v1 signs, WearGait-PD           | P3         | CONFIRMED     | v1 signs alone 0.551-0.764 vs lowest expert support 0.970 (lie AUROC)                                                                                       |
| raw-zero watch, MMAct           | N1         | NOT CONFIRMED | doing nothing: raw 0 68.7% vs mean fill 55.9%                                                                                                               |
| raw-zero watch, MMAct           | N2         | NOT CONFIRMED | REV3 flags 90.4% (mean fill 91.4%), names the watch on 45.7% of flagged (mean fill 48.7%)                                                                   |
| raw-zero watch, MMAct           | N3         | NOT CONFIRMED | REV3 use / drop 53.6% vs doing nothing 68.7% (raw 0)                                                                                                        |
| raw-zero watch, MMAct           | N4         | CONFIRMED     | REV3 AUROC, mean of 4 people: raw 0 0.949 vs mean fill 0.945                                                                                                |
| partial IMU-off, WearGait-PD    | I1         | CONFIRMED     | legs k=8 gain: p25 -0.21, p50 +1.06, p75 +7.85, p100 +39.52                                                                                                 |
| partial IMU-off, WearGait-PD    | I2         | CONFIRMED     | legs k=8 doing nothing: p25 95.6% vs whole clip 52.5%                                                                                                       |
| partial IMU-off, WearGait-PD    | I3         | CONFIRMED     | legs k=8 REV3 AUROC: p25 0.647, p50 0.730, p75 0.824, p100 0.912                                                                                            |
| partial IMU-off, WearGait-PD    | I4         | CONFIRMED     | k=1 gain: arms p25 -0.24, arms p50 -0.28, arms p75 -0.42, trunk p25 -0.23, trunk p50 -0.38, trunk p75 -0.64, legs p25 -0.17, legs p50 -0.21, legs p75 -0.22 |
| stuck + saturation, UTD-MHAD    | F1         | CONFIRMED     | harmful-fault AUROC: confidence alone 0.922 vs REV3 0.900                                                                                                   |
| stuck + saturation, UTD-MHAD    | F2         | CONFIRMED     | names the faulty sensor (harmful): REV3 70.5% vs TMC alone 17.0%                                                                                            |
| stuck + saturation, UTD-MHAD    | F3         | CONFIRMED     | share of answers changed by level: f_stuck 1.4% < 6.9% < 29.5%; f_sat 1.5% < 2.1% < 3.1%                                                                    |
| stuck + saturation, UTD-MHAD    | F4         | CONFIRMED     | saturation level 1 changes 1.49% of answers (predicted < 5%)                                                                                                |
| stuck + saturation, MMAct       | F1         | CONFIRMED     | harmful-fault AUROC: confidence alone 0.883 vs REV3 0.876                                                                                                   |
| stuck + saturation, MMAct       | F2         | CONFIRMED     | names the faulty sensor (harmful): REV3 66.3% vs TMC_S3 alone 11.4%                                                                                         |
| stuck + saturation, MMAct       | F3         | CONFIRMED     | share of answers changed by level: f_stuck 3.3% < 10.4% < 32.4%; f_sat 5.1% < 12.4% < 18.5%                                                                 |
| stuck + saturation, MMAct       | F4         | NOT CONFIRMED | saturation level 1 changes 5.13% of answers (predicted < 5%)                                                                                                |
| stuck + saturation, WearGait-PD | F1         | NOT CONFIRMED | harmful-fault AUROC: confidence alone 0.909 vs REV3 0.941                                                                                                   |
| stuck + saturation, WearGait-PD | F2         | CONFIRMED     | names the faulty sensor (harmful): REV3 82.0% vs TMC_W alone 56.1%                                                                                          |
| stuck + saturation, WearGait-PD | F3         | CONFIRMED     | share of answers changed by level: f_stuck 0.9% < 2.0% < 8.4%; f_sat 0.0% < 0.0% < 0.1%                                                                     |
| stuck + saturation, WearGait-PD | F4         | CONFIRMED     | saturation level 1 changes 0.00% of answers (predicted < 5%)                                                                                                |

## Checks

2658 numbers compared with the scorers' own logs; 0 disagree.

