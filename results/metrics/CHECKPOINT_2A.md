# Checkpoint 2A: non-model length audit

The frozen sample and its fixed split were not modified.

## Dataset length distributions

```json
{
  "bench_deploy": {
    "frozen_300": {
      "max": 650,
      "mean": 265.76,
      "median": 242.5,
      "min": 106,
      "n": 300,
      "p05": 136.95,
      "p25": 196.75,
      "p75": 319.0,
      "p95": 469.1,
      "std_population": 100.06845523606994
    },
    "full_valid_pool": {
      "max": 672,
      "mean": 264.2512019230769,
      "median": 238.0,
      "min": 103,
      "n": 832,
      "p05": 137.0,
      "p25": 191.0,
      "p75": 323.25,
      "p95": 467.89999999999986,
      "std_population": 100.36279441786718
    }
  },
  "bench_eval": {
    "frozen_300": {
      "max": 933,
      "mean": 264.26,
      "median": 239.5,
      "min": 80,
      "n": 300,
      "p05": 128.85,
      "p25": 186.0,
      "p75": 315.5,
      "p95": 472.15000000000003,
      "std_population": 111.60785097832499
    },
    "full_valid_pool": {
      "max": 933,
      "mean": 266.3556797020484,
      "median": 245.0,
      "min": 68,
      "n": 1074,
      "p05": 122.65,
      "p25": 180.0,
      "p75": 321.75,
      "p95": 489.0,
      "std_population": 117.79355789482396
    }
  },
  "casual_deploy": {
    "frozen_300": {
      "max": 1139,
      "mean": 299.21,
      "median": 230.5,
      "min": 52,
      "n": 300,
      "p05": 94.80000000000001,
      "p25": 155.5,
      "p75": 390.25,
      "p95": 718.2500000000007,
      "std_population": 203.53151901691623
    },
    "full_valid_pool": {
      "max": 1139,
      "mean": 285.5812854442344,
      "median": 222.5,
      "min": 52,
      "n": 1058,
      "p05": 81.85,
      "p25": 149.0,
      "p75": 363.75,
      "p95": 712.15,
      "std_population": 194.85217000685842
    }
  },
  "casual_eval": {
    "frozen_300": {
      "max": 711,
      "mean": 110.71333333333334,
      "median": 92.0,
      "min": 5,
      "n": 300,
      "p05": 29.0,
      "p25": 50.0,
      "p75": 148.25,
      "p95": 269.0,
      "std_population": 82.24648617958636
    },
    "full_valid_pool": {
      "max": 711,
      "mean": 108.66224489795918,
      "median": 91.0,
      "min": 5,
      "n": 980,
      "p05": 29.0,
      "p25": 54.0,
      "p75": 146.0,
      "p95": 243.0,
      "std_population": 76.33241750598086
    }
  }
}
```

## Upstream matching provenance

The upstream author intended casual_deploy_matched to match bench_eval, not casual_eval. Matching used character lengths of formatted bench_eval prompts and ShareGPT first turns.

- Code revision: `b6a5000cba8cf99d9a7e313f6371b589d9761b05`
- Code: https://github.com/viliana-dev/eval-awareness-format/blob/b6a5000cba8cf99d9a7e313f6371b589d9761b05/scripts/create_length_matched.py

## Length-only classifier

```json
{
  "accuracy_at_default_threshold": 0.7222222222222222,
  "auroc": 0.8085956790123456,
  "features": [
    "character_count",
    "whitespace_word_count",
    "dataset_length_field"
  ],
  "fit_rows": 840,
  "intercept": -0.059105632948264465,
  "model": "StandardScaler + L2 LogisticRegression(C=1.0)",
  "note": "The dataset length field exactly equals Python character count in all full valid pools, so two feature columns are duplicates.",
  "positive_label": "evaluation (context_label=1)",
  "standardized_coefficients_in_feature_order": [
    -1.8980388457713726,
    3.013232823703321,
    -1.8980388457713726
  ],
  "test_class_counts": {
    "0": 180,
    "1": 180
  },
  "test_rows": 360
}
```

## Integrity

```json
{
  "dataset_length_equals_character_count_for_all_valid_pool_rows": true,
  "frozen_rows_unchanged": true,
  "test_rows": 360,
  "train_rows": 840,
  "unique_stable_ids": 1200
}
```
