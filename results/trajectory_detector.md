# IQA-SSC Active Trajectory Detector

Status: `computed`

## Validity

| Family | Valid | Total | Coverage |
|---|---:|---:|---:|
| bilateral | 2500 | 2500 | 1.0000 |
| jpeg | 2500 | 2500 | 1.0000 |
| gaussian_blur | 2500 | 2500 | 1.0000 |

## Comparisons

| Comparison | Stratum | N positive | N negative | AUC | CI lower | CI upper | Gate |
|---|---|---:|---:|---:|---:|---:|:---:|
| bilateral_vs_jpeg | full | 1250 | 1250 | 0.996111 | 0.994625 | 0.997434 | PASS |
| bilateral_vs_jpeg | severe | 500 | 500 | 0.999912 | 0.999760 | 0.999996 | PASS |
| bilateral_vs_jpeg | mild | 500 | 500 | 0.983732 | 0.976500 | 0.989752 | PASS |
| bilateral_vs_blur | full | 1250 | 1250 | 0.965181 | 0.958234 | 0.971507 | PASS |
| bilateral_vs_blur | severe | 500 | 500 | 0.999568 | 0.998944 | 0.999952 | PASS |
| bilateral_vs_blur | mild | 500 | 500 | 0.945728 | 0.932396 | 0.958233 | PASS |

## Ablations (full stratum)

| Feature set | Comparison | AUC | CI lower | CI upper |
|---|---|---:|---:|---:|
| jpeg_probe_6 | bilateral_vs_jpeg | 0.991112 | 0.988152 | 0.993750 |
| jpeg_probe_6 | bilateral_vs_blur | 0.910934 | 0.899383 | 0.921505 |
| blur_probe_6 | bilateral_vs_jpeg | 0.917489 | 0.906510 | 0.927810 |
| blur_probe_6 | bilateral_vs_blur | 0.933334 | 0.923719 | 0.942642 |
| goc_only_6 | bilateral_vs_jpeg | 0.987754 | 0.984269 | 0.990795 |
| goc_only_6 | bilateral_vs_blur | 0.957811 | 0.950135 | 0.964991 |
| s_grid_only_6 | bilateral_vs_jpeg | 0.952077 | 0.943192 | 0.960813 |
| s_grid_only_6 | bilateral_vs_blur | 0.710415 | 0.690681 | 0.730375 |
| single_point_q30_sigma3 | bilateral_vs_jpeg | 0.987940 | 0.984482 | 0.990987 |
| single_point_q30_sigma3 | bilateral_vs_blur | 0.968310 | 0.962092 | 0.974101 |

Primary detector gate: **PASS**

