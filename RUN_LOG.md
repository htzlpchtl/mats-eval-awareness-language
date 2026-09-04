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

## 2026-09-04 — Stage 2 frozen English dataset and split

- Command: `HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 .venv/bin/python -m src.prepare_data`.
- Reproducibility command: `HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 .venv/bin/python -m src.prepare_data --output /private/tmp/mats_stage2_repro.jsonl --manifest /private/tmp/mats_stage2_repro_manifest.json --checkpoint-report /private/tmp/mats_stage2_repro.md`.
- Seed: 42.
- Dataset revision: `a50e4c983e7e66ebbe8f160e3549cc98e29271d4`.
- Manual exclusions: bench_deploy original rows 259 and 735;
  casual_deploy original rows 559 and 651.
- Exact deduplication retained the lowest original row index in each quadrant;
  115 later duplicate rows were removed from the eligible pools.
- Sample size: 1,200 total; exactly 300 per quadrant.
- Split: 840 train and 360 test; exactly 210/90 per quadrant.
- CV: five folds; exactly 42 training examples from every quadrant per fold.
- Files written: `data/selected/english_selected.jsonl`,
  `data/selected/english_selected_manifest.json`, and
  `data/selected/CHECKPOINT_2.md`.
- Frozen JSONL SHA-256:
  `a6b7b1fa482d5ad971a2a68432dcffa6b96bfc48f776646fcc782c5df422eef8`.
- Checks: independently traced every frozen row to its pinned source row;
  verified all labels, IDs, exclusions, uniqueness, split and fold counts;
  reproduced an identical JSONL in a second offline run; 14 tests passed with
  warnings treated as errors.
- Error caught and fixed during validation: an input prompt contains a Unicode
  line-separator character that `str.splitlines()` treated as a JSONL boundary.
  Machine-readable output now escapes non-ASCII code points; decoded prompt
  text and stable text hashes remain unchanged.

## 2026-09-04 — Checkpoint 2A non-model length audit

- Command: `HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 .venv/bin/python -m src.audit_length`.
- Reproducibility command: `HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 .venv/bin/python -m src.audit_length --output /private/tmp/mats_2a_repro.json --report /private/tmp/mats_2a_repro.md`.
- Dataset revision: `a50e4c983e7e66ebbe8f160e3549cc98e29271d4`.
- Frozen JSONL SHA-256 verified unchanged:
  `a6b7b1fa482d5ad971a2a68432dcffa6b96bfc48f776646fcc782c5df422eef8`.
- Full valid pools mean dataset lengths: bench_eval 266.36;
  casual_eval 108.66; bench_deploy 264.25; casual_deploy 285.58.
- Frozen-sample mean dataset lengths: bench_eval 264.26; casual_eval
  110.71; bench_deploy 265.76; casual_deploy 299.21.
- Upstream code inspected at `viliana-dev/eval-awareness-format` revision
  `b6a5000cba8cf99d9a7e313f6371b589d9761b05`.
- Upstream matching finding: `casual_deploy_matched` was matched to
  formatted `bench_eval` character lengths using 30 percentile-derived bins,
  sampling up to the bench count per bin with seed 42.
- Length-only probe: StandardScaler plus L2 LogisticRegression (`C=1.0`),
  trained on the fixed 840 training rows using character count, whitespace
  word count, and the source dataset length field; evaluated once on the fixed
  360 test rows (180 per class).
- Test AUROC: 0.8085956790123456.
- Test accuracy at the default threshold: 0.7222222222222222.
- The dataset length field exactly equals Python character count in every row
  of all four valid pools, so two classifier feature columns are identical.
- Files written: `results/metrics/checkpoint2a_length_audit.json` and
  `results/metrics/CHECKPOINT_2A.md`.
- Checks: a second offline run produced byte-identical reports; artifact and
  frozen-file hash validation passed; 16 tests passed with warnings treated as
  errors. The frozen sample, split, and manifest were not modified.

## 2026-09-04 — Checkpoint 2B pairwise length-matched v2 dataset

- Command: `HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 .venv/bin/python -m src.prepare_data_v2`.
- Seed: 42.
- Dataset revision: `a50e4c983e7e66ebbe8f160e3549cc98e29271d4`.
- The v1 pilot artifacts were retained byte-unchanged and marked retired.
- Matching was performed independently for benchmark and casual formats.
  Candidate pools were restricted to overlapping inclusive character-length
  support; 300 evaluation anchors were sampled; character and whitespace-word
  counts were standardized over all common-support eval and deploy candidates;
  SciPy `linear_sum_assignment` produced a globally minimum-cost unique deploy
  assignment.
- Common-support availability: benchmark eval 1,052 / deploy 832, range
  103–672 characters; casual eval 748 / deploy 1,004, range 52–711.
- Output: 600 matched pairs / 1,200 rows, exactly 300 per quadrant.
- Pair-preserving split per format: 210 train / 90 test pairs. Every training
  fold contains exactly 42 benchmark pairs and 42 casual pairs.
- V2 JSONL SHA-256:
  `4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639`.
- Training-only length classifier fold AUROCs: 0.6624149659863945,
  0.6383219954648527, 0.6873582766439909, 0.6352749433106576,
  0.6534863945578231; mean 0.6553713151927438.
- No classifier was fit or evaluated on v2 test rows. Test rows were only
  assigned and checked for pair/split/fold integrity.
- Matching distances: benchmark mean 0.06765762615706201, median
  0.05305297731013649, max 0.6965162000096754; casual mean
  0.19025652832856357, median 0.1823498300175424, max 0.78004631324381.
- Poor-match diagnostic rule: distance greater than Q3 + 1.5×IQR. Flagged 19
  benchmark and 12 casual pairs; all are listed in the report.
- Files written: `data/selected/english_selected_v2.jsonl`,
  `data/selected/english_selected_v2_manifest.json`,
  `data/selected/PILOT_V1_RETIRED.md`,
  `results/metrics/checkpoint2b_matching_audit.json`, and
  `results/metrics/CHECKPOINT_2B.md`.
- Checks: reproduced byte-identical v2 JSONL; independently reconstructed both
  rectangular cost matrices and verified observed assignments equal SciPy's
  global optimum; verified source exclusions, unique prompts and deploy rows,
  paired split/fold invariants, and unchanged v1 SHA; 19 tests passed with
  warnings treated as errors.

## 2026-09-04 — Checkpoint 3 compute handoff

- Human approved continuing under time constraints with the v2 matched dataset.
  The residual training-only length-probe mean CV AUROC of
  `0.6553713151927438` remains a documented limitation; the v2 final test
  partition remains uninspected by classifiers.
- Prepared `docs/COMPUTE_SETUP.md`; no compute was provisioned and no model
  weights were downloaded or loaded.
- Initial target: one NVIDIA A40 48 GB or RTX A6000 48 GB, approximately 80 GB
  or more usable persistent storage, Linux, and SSH access.
- Repository remote:
  `https://github.com/htzlpchtl/mats-eval-awareness-language.git`.
- Frozen v2 SHA-256 carried into the handoff:
  `4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639`.
- Model: `Qwen/Qwen3.5-9B`; Hugging Face revision queried on 2026-09-04:
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- The official model-card loading path uses `AutoProcessor` and
  `AutoModelForMultimodalLM`. The handoff checks that these classes, CUDA, and
  BF16 support are available before Stage 4.

## 2026-09-04 — Checkpoint 4 Qwen GPU smoke test

- Final command:
  `HF_HOME=/workspace/.cache/huggingface .venv/bin/python -m src.model_smoke_test --config config.yaml --dataset data/selected/english_selected_v2.jsonl --dataset-sha256 4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639 --model Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Validation commands:
  - `.venv/bin/python -m py_compile src/model_smoke_test.py src/extract_activations.py`
  - `.venv/bin/python -m pytest -q -W error`
  - `.venv/bin/python -m pip check`
  - `sha256sum data/selected/english_selected_v2.jsonl artifacts/activations/stage4_smoke.npz`
  - `.venv/bin/python -m pip freeze --all > results/metrics/stage4_environment.txt`
- Seed: 42. Smoke sample: four training rows, one per quadrant. Throughput
  sample: 16 training rows, four character-length quantiles per quadrant.
- Frozen v2 path and SHA-256:
  `data/selected/english_selected_v2.jsonl`,
  `4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639`.
  The file was checked before and after the model run and was not modified.
- Model: `Qwen/Qwen3.5-9B`; requested and resolved revision:
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; loaded class:
  `Qwen3_5ForCausalLM`; 8,953,803,264 parameters; BF16 on `cuda:0`;
  32 language blocks; hidden size 4096. Loading information contained zero
  missing, mismatched, unexpected, or error entries.
- Environment: Python 3.12.3; PyTorch 2.14.0+cu130; Transformers 5.16.1;
  datasets 4.8.5; Accelerate 1.14.0; huggingface_hub 1.30.0; PyYAML 6.0.3;
  CUDA runtime 13.0; cuDNN 92400; NVIDIA A40; driver 580.159.04; BF16
  supported. The isolated environment cannot import `torchaudio`.
- The live hidden-state API returned 33 BF16 tensors of shape
  `(4, 166, 4096)`. Index 0 exactly matched embeddings; indices 1–31
  exactly matched raw blocks 1–31; index 32 exactly matched the terminal
  RMSNorm output and did not match raw block 32. The stored activations
  therefore use forward hooks for the raw outputs of all 32 blocks.
- Dynamic right padding was verified manually and by assertions. For all four
  smoke prompts, the final non-padding token was token ID 198 (decoded newline)
  after `<think>`; final indices were 44, 33, 83, and 165.
- Fixed downstream extraction policy: batch size 4; frozen-manifest order;
  consecutive-row batches; dynamic right padding; no truncation; final token
  selected as the rightmost nonzero attention-mask position; raw outputs of all
  32 decoder blocks; CPU storage as float32. English, Spanish, and Japanese
  must use this identical path and policy.
- Stored activation shape: `(4, 32, 4096)`; all values finite. Artifact
  `artifacts/activations/stage4_smoke.npz` reloaded with IDs, quadrants,
  shape, and values exactly unchanged. Artifact SHA-256:
  `0f32ae1c528a5f66a7c78771f9a12a92a86b0cd2a16549520113e2b3a48bb5ca`.
- Batched versus single-example extraction passed the recorded numerical
  agreement criterion: minimum layer cosine 0.9993124008178711 and maximum
  per-layer relative L2 0.037078216671943665. Strict elementwise
  `rtol=0.02, atol=0.02` all-close was false; maximum absolute difference
  was 1.5 and mean absolute difference was 0.017855897545814514. This BF16
  batch-shape dependence motivated freezing the batching policy.
- GPU allocated-memory peak: 17,427.6318359375 MiB during the smoke forward.
  The representative throughput benchmark peaked at 17,308.7275390625 MiB
  allocated and 17,340 MiB reserved.
- Throughput: 16 prompts and 893 non-padding tokens in 0.7397667160257697
  seconds; 21.628439957337363 prompts/s and 1207.1373051188916 non-padding
  tokens/s. Projected 1,200-prompt time: 55.48250370193273 seconds
  (0.9247083950322121 minutes), including GPU-to-CPU float32 transfer and
  excluding model load.
- Tracked reporting files: compact aggregate-only
  `results/metrics/stage4_summary.json` and package manifest
  `results/metrics/stage4_environment.txt`.
- The full `results/metrics/stage4_smoke.json` remains local and unstaged for
  audit because it contains raw prompts, rendered templates, token IDs, and
  attention masks. The activation artifact
  `artifacts/activations/stage4_smoke.npz` is also local and ignored.
- Checks: 19 tests passed with warnings treated as errors; `pip check`
  reported no broken requirements; syntax and diff checks passed.
- Issues fixed during Stage 4: isolated the venv from incompatible system
  multimedia packages; corrected chat-template handling for a returned
  `BatchEncoding`; diagnosed the live 33-state mapping; and made set-valued
  loading metadata JSON-serializable. `torchaudio` was never installed.
  `torchvision==0.29.0` was installed earlier during compatibility work and
  remains in the isolated environment, but the smoke test does not import or
  use it.
- Warning: Hugging Face Hub access was unauthenticated. No Stage 5 extraction
  or full-dataset activation extraction was run.
