# Paper-ready tables

## Table 1 — Dataset

| site | management | start_date | end_date | rows | growing_QC_ge_075 |
| --- | --- | --- | --- | --- | --- |
| US-Ne2 | irrigated | 2001-01-01 | 2013-12-31 | 4748 | 1491 |
| US-Ne3 | rainfed | 2001-01-01 | 2013-12-31 | 4748 | 1507 |

## Table 2 — Main RMSE results

| target_site | model | V0_RMSE | V1_RMSE | V2_RMSE | V3_RMSE | V3_minus_V2_95CI | V3_minus_V1_95CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| US-Ne2 | climatology | 34.36 | 35.46 | 36.17 | 36.96 | 0.62 [0.27, 0.73] | 1.09 [-1.98, 2.96] |
| US-Ne2 | random_forest | 22.98 | 24.35 | 23.53 | 25.63 | 2.26 [1.82, 3.57] | 1.71 [0.32, 2.15] |
| US-Ne2 | ridge | 27.49 | 29.08 | 28.49 | 29.14 | 0.44 [0.17, 0.64] | 0.05 [-0.53, 0.77] |
| US-Ne3 | climatology | 28.43 | 28.73 | 29.56 | 29.82 | 0.36 [0.09, 1.00] | 1.09 [-0.60, 3.10] |
| US-Ne3 | random_forest | 20.01 | 21.92 | 21.02 | 23.25 | 2.58 [1.67, 3.13] | 2.02 [-0.02, 2.39] |
| US-Ne3 | ridge | 24.58 | 25.7 | 25.05 | 25.88 | 0.51 [0.27, 0.95] | 0.28 [-0.31, 1.01] |
