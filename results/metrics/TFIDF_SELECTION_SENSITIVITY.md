# TF-IDF selection-sensitivity diagnostic

All results use training rows only. Each TF-IDF vectorizer and logistic classifier was fit inside its CV training fold.

| Dataset | Mean CV AUROC | Range |
|---|---:|---:|
| Frozen v2 pooled | 0.895692 | 0.875142-0.920918 across folds |
| Unmatched source-pool samples (20 seeds) | 0.913403 | 0.898044-0.931604 across seed means |
| Frozen v2 benchmark only | 0.886168 | 0.857143-0.911565 across folds |
| Frozen v2 casual only | 0.963379 | 0.948413-0.979025 across folds |

Frozen-v2 minus unmatched mean: -0.017711 AUROC.
Fraction of unmatched seed means at least as high as frozen v2: 1.000.

The unmatched samples were independently drawn per quadrant after exact deduplication/manual exclusions and restriction to the same benchmark and casual common character-length supports used by v2. Frozen held-out IDs were excluded before sampling. No pairwise matching was used.
