# Run log

Record experimental runs factually. Do not record secrets.

For each run include:

- timestamp and exact command;
- relevant configuration and random seed;
- Python, PyTorch, Transformers, CUDA, and GPU versions where relevant;
- model ID and revision;
- dataset revision or fingerprint;
- sample size;
- files written;
- metrics;
- warnings and errors.

## 2026-09-04 — Stage 1 dataset inspection

- Commands:
  - `HF_HOME=.cache/huggingface .venv/bin/python -c 'from datasets import get_dataset_config_names; ...'`
  - `HF_HOME=.cache/huggingface .venv/bin/python -c 'from huggingface_hub import HfApi; from datasets import load_dataset; ...'`
  - `HF_HOME=.cache/huggingface .venv/bin/python -m src.inspect_data --output results/metrics/stage1_dataset_inspection.json`
  - `.venv/bin/python -m pytest -q -W error`
- Seed: 42.
- Python: 3.12.2; datasets: 4.8.5; PyArrow: 25.0.1; NumPy: 2.5.2.
- Dataset: `viliana-dev/eval-awareness-2x2`.
- Dataset revision: `a50e4c983e7e66ebbe8f160e3549cc98e29271d4`.
- Rows inspected: bench_eval 1,076; casual_eval_mutual 1,076;
  bench_deploy_rewritten 835; casual_deploy_matched 1,076.
- Files written: `results/metrics/stage1_dataset_inspection.json` and
  `results/metrics/CHECKPOINT_1.md`.
- Checks: 10 tests passed with warnings treated as errors.
- Integrity counts: 0 missing/empty candidate prompts; 110 within-quadrant
  exact-duplicate groups (225 duplicate rows; 115 excess rows); 0
  cross-quadrant exact-prompt duplicate groups; 0 structurally malformed rows;
  4 non-Latin-script rows flagged for review. No rows were excluded.
- Warning: requests to the Hugging Face Hub were unauthenticated.
- Error encountered: the installed datasets API reported only a `default`
  configuration; loading it failed because the quadrant Parquet schemas differ.
  The inspection therefore downloaded and read the four declared Parquet files
  directly, pinned to the dataset revision above.
