# Minimal multilingual transfer checkpoint

> Probe B is source-confounded and is not a clean measure of evaluation awareness.

| Transfer | AUROC | Accuracy | Mean score | Predicted eval rate | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| EN->EN | 0.997284 | 0.966667 | -0.193214 | 0.527778 | 0.023970 | 0.025188 |
| EN->ES | 0.881265 | 0.575000 | -7.312260 | 0.075000 | 0.366878 | 0.403291 |
| EN->JA | 0.864259 | 0.500000 | -11.162112 | 0.000000 | 0.496966 | 0.498921 |

## EN->EN quadrant breakdown

| Quadrant | n | Mean score | Median | SD | Eval-class rate | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| bench_eval | 90 | 9.283227 | 9.830058 | 3.753744 | 0.988889 | 0.988889 |
| casual_eval | 90 | 13.941226 | 14.996951 | 4.034198 | 1.000000 | 1.000000 |
| bench_deploy | 90 | -8.206866 | -9.208462 | 5.383582 | 0.122222 | 0.877778 |
| casual_deploy | 90 | -15.790444 | -14.670347 | 6.774826 | 0.000000 | 1.000000 |

## EN->ES quadrant breakdown

| Quadrant | n | Mean score | Median | SD | Eval-class rate | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| bench_eval | 90 | -4.815078 | -3.542657 | 4.254589 | 0.066667 | 0.066667 |
| casual_eval | 90 | -2.259615 | -2.236989 | 3.458398 | 0.233333 | 0.233333 |
| bench_deploy | 90 | -6.899399 | -6.622778 | 2.487982 | 0.000000 | 1.000000 |
| casual_deploy | 90 | -15.274949 | -15.314703 | 4.489084 | 0.000000 | 1.000000 |

## EN->JA quadrant breakdown

| Quadrant | n | Mean score | Median | SD | Eval-class rate | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| bench_eval | 90 | -9.182027 | -9.410661 | 1.832259 | 0.000000 | 0.000000 |
| casual_eval | 90 | -6.929648 | -7.151539 | 2.005650 | 0.000000 | 0.000000 |
| bench_deploy | 90 | -10.414749 | -10.030705 | 2.098189 | 0.000000 | 1.000000 |
| casual_deploy | 90 | -18.122025 | -16.925698 | 6.651918 | 0.000000 | 1.000000 |
