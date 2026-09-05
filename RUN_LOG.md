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

## 2026-09-05 — Checkpoint 5 full English activation extraction

- Extraction command: HF_HOME=/workspace/.cache/huggingface
  .venv/bin/python -m src.extract_activations --config config.yaml --dataset
  data/selected/english_selected_v2.jsonl --dataset-sha256
  4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639
  --model Qwen/Qwen3.5-9B --revision
  c202236235762e1c871ad0ccb60c8ee5ba337b9a.
- Recovery-audit command: HF_HOME=/workspace/.cache/huggingface
  .venv/bin/python -m src.audit_stage5_artifact --dataset
  data/selected/english_selected_v2.jsonl --dataset-sha256
  4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639
  --artifact artifacts/activations/stage5_english_v2.npz --manifest
  results/metrics/stage5_english_manifest.json --report
  results/metrics/stage5_summary.json --observed-live-device-memory-mib
  17846.
- Seed: 42. Language: English. Frozen input:
  data/selected/english_selected_v2.jsonl; SHA-256 was verified before the
  preflight, immediately before the full extraction, after extraction, and
  during recovery as
  4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639.
  The frozen file was not modified or resampled.
- Model: Qwen/Qwen3.5-9B; requested and resolved revision
  c202236235762e1c871ad0ccb60c8ee5ba337b9a; loaded class
  Qwen3_5ForCausalLM; BF16 on the NVIDIA A40; 32 blocks; hidden size 4096.
- The required 40-row preflight used 10 evenly spaced manifest positions per
  quadrant. Its (40, 32, 4096) shape, hook coverage, final-token indexing,
  and finite-value assertions passed before the full run began.
- Full extraction processed the frozen manifest once: 1,200 prompts in 300
  consecutive-row batches of four, with dynamic right padding, no truncation,
  no generation, and all 32 raw block outputs captured by forward hooks.
  Activations were selected at the rightmost non-padding token and stored as
  CPU float32.
- Qwen input-token lengths over all 1,200 prompts: minimum 18, median 54,
  95th percentile 107, maximum 359. Zero exceeded the checked 262,144-token
  model limit.
- Four ignored recovery checkpoints were written and reloaded, one per
  contiguous 300-row quadrant:
  artifacts/activations/stage5_english_01_bench_deploy.npz
  (038cbed29f42fcb9bb269737458d3bdcfd9e23444dbe115741e16a54d92fad0a),
  stage5_english_02_bench_eval.npz
  (8c2febe10a57d492653ef28fabc1ba02d771306faaab0cad93b2538a07e7cf20),
  stage5_english_03_casual_deploy.npz
  (3cc22d17b478d5e76dc916bd67f1b827aeeee41a32322826c58ec739b0f99895),
  and stage5_english_04_casual_eval.npz
  (3629a2659c203825b742782b9b8124cd4928b56d0d8575ecc7706b5c5984c4fa).
- Final ignored artifact:
  artifacts/activations/stage5_english_v2.npz; shape (1200, 32, 4096);
  float32; 292,374,412 bytes; SHA-256
  dbac459e5670ee9e61fa90c7e62f4e60ce305589b6d1c55ac2b041c7c90136f8.
  It embeds global row indices, stable IDs, quadrants, splits, CV folds,
  context and format labels, pair IDs, one-based layer numbers, zero-based
  module indices, and raw-hook source strings.
- Independent recovery audit verified every metadata array against frozen
  JSONL order, every manifest row and all 32 layer mappings, all-finite
  values, and exact equality between every quadrant checkpoint activation
  array and its final-artifact slice.
- Tracked files written: results/metrics/stage5_english_manifest.json and
  results/metrics/stage5_summary.json.
- Implementation issue: the extraction process completed final artifact
  save/reload verification and wrote the manifest, then exited while
  serialising the summary because set-valued Transformers loading metadata
  lacked the Stage 4 JSON fallback. The extractor now uses a tested set- and
  NumPy-aware serializer. The full extraction was intentionally not rerun.
- Because the summary failure occurred before telemetry was persisted, exact
  Stage 5 extraction runtime, prompts/second, and PyTorch peak VRAM are
  unavailable. During the live run, nvidia-smi observed 17,846 MiB used by
  the process. Same-policy Stage 4 peak allocated memory was 17,427.632 MiB.
  Second-resolution file intervals retained in the recovery report are not
  presented as exact extraction timing.
- Warning: Hugging Face Hub access was unauthenticated.
- Test history: the first post-implementation run had one overly specific
  test error-message match fail and 22 tests pass; the test was corrected.
  The suite then passed 23 tests. After adding the serializer regression,
  the final suite passed 24 tests with warnings treated as errors.
- No Probe A training or other Stage 6 work was started.
## 2026-09-05 — Checkpoint 7 English Probe A and Probe B

- Command: `HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1
  .venv/bin/python -m src.run_english_probes --dataset
  data/selected/english_selected_v2.jsonl --activations
  artifacts/activations/stage5_english_v2.npz --dataset-sha256
  4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639
  --activation-sha256
  dbac459e5670ee9e61fa90c7e62f4e60ce305589b6d1c55ac2b041c7c90136f8`.
- Inputs were hash-checked and artifact row/layer metadata was verified before
  use. Environment: Python 3.12.3, NumPy 2.5.2, scikit-learn 1.9.0. Seed 42.
- Both probes used a `StandardScaler` plus L2 `LogisticRegression` pipeline
  (`C=1.0`, `max_iter=2000`, `random_state=42`). A fresh whole pipeline was
  fit inside each training fold. No convergence warnings occurred.
- Probe A trained on 210 bench-eval and 210 casual-deploy training rows. All
  32 layers had mean five-fold CV AUROC 1.0, so the fixed first-argmax rule
  selected transformer block 1. Its held-out in-distribution AUROC/accuracy
  were 1.0/1.0. Without refitting, its crossed-format AUROC was 0.0 and
  accuracy was 0.03333333333333333.
- Probe B used all 840 training rows with exactly 42 examples from every
  quadrant in every validation fold. Block 32 was selected from training-only
  CV: mean AUROC 0.9966836734693878; fold AUROCs 0.9987244897959184,
  0.9995748299319729, 0.9912131519274376, 0.9953231292517006, and
  0.9985827664399093.
- The frozen final Probe B achieved held-out English AUROC
  0.997283950617284 and accuracy 0.9666666666666667, with confusion matrix
  `[[169, 11], [1, 179]]` in label order `[0, 1]`.
- Held-out quadrant mean scores and correct proportions: bench-eval 9.283227
  / 0.988889; casual-eval 13.941226 / 1.0; bench-deploy -8.206866 / 0.877778;
  casual-deploy -15.790444 / 1.0.
- Twenty deterministic shuffled-label fits (seeds 42–61) had mean held-out
  AUROC 0.49794598765432097, population SD 0.03816746643560377, and range
  0.4356172839506173–0.5461111111111111. The character-count plus Qwen-token-
  count baseline had held-out AUROC 0.6411265432098766 and accuracy
  0.6027777777777777.
- The combined runner took 231.98357144976035 seconds. Probe A reporting time
  was 26.295414465945214 seconds and Probe B reporting time was
  127.27075225999579 seconds; these nested times exclude initial shared input
  loading and some combined-run overhead.
- Per-fold/layer metrics, every held-out prediction, all shuffled and length
  predictions, fitted frozen pipelines, and three figures were saved. An
  independent path recomputed layer argmaxes, AUROCs, ID coverage, and exact
  fitted-pipeline scores from the activation artifact.

## 2026-09-05 — Source-confounding diagnostics before multilingual decision

- Command: `HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1
  .venv/bin/python -m src.run_source_diagnostics --dataset
  data/selected/english_selected_v2.jsonl --activations
  artifacts/activations/stage5_english_v2.npz --dataset-sha256
  4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639
  --activation-sha256
  dbac459e5670ee9e61fa90c7e62f4e60ce305589b6d1c55ac2b041c7c90136f8`.
- The raw model-facing English text baseline used default word-unigram
  `TfidfVectorizer` plus the fixed L2 logistic classifier. Both vectorizer and
  classifier were fit separately inside every training fold. Fold AUROCs were
  0.9209183673469389, 0.8942743764172335, 0.8884637188208616,
  0.8996598639455783, and 0.875141723356009; mean 0.8956916099773242.
- Benchmark-to-casual selected block 13 using only the 420 benchmark training
  rows (mean CV AUROC 0.9904761904761905), then achieved AUROC
  0.7986394557823128 on the 420 casual training rows. Fixed-threshold accuracy
  was 0.5095238095238095 because almost all scores shifted below zero.
- Casual-to-benchmark selected block 1 using only the 420 casual training rows
  (mean CV AUROC 1.0), then achieved AUROC 0.6571655328798185 on the 420
  benchmark training rows. Fixed-threshold accuracy was 0.5 because every
  benchmark row scored below zero.
- No held-out test rows were used by either new diagnostic. Independent saved-
  file verification confirmed exact source/target ID sets, selected-layer
  argmaxes, AUROCs, and pipeline decision scores. No convergence warnings
  occurred. Diagnostic runtime was 89.59300431190059 seconds.
- An initial ad hoc independent verifier incorrectly requested within-quadrant
  AUROCs for single-class groups and compared Probe A scores under a different
  matrix batch shape; it failed without changing results. The corrected path
  uses quadrant score summaries and original evaluation groupings and passed.
- Files: `results/metrics/english_go_no_go_summary.json`,
  `results/metrics/ENGLISH_GO_NO_GO.md`, probe-specific JSON/CSV reports and
  predictions, source-diagnostic JSON/CSVs, figures, and local fitted pipeline
  artifacts. Translation and optional geometry were not started.

## 2026-09-05 - Minimal held-out multilingual transfer checkpoint

- Translated only the 360 frozen test rows into Spanish and Japanese with
  `deepseek/deepseek-v4-flash-0731` through OpenRouter, temperature 0, seed 42,
  reasoning excluded, stable IDs/order/split preserved, and resumable ignored
  per-ID/language caching. The initial 720 calls took 108.619 seconds. Focused
  retries corrected instruction-following, Chinese-instead-of-Japanese, omitted
  structure, and wrapper-output failures; no training rows were translated.
- Final translation SHA-256: Spanish
  `209acb342096382a2034ee06d7ccbf6823c931e2982e1cc7bb91f6cf74b8456b`;
  Japanese `52a4edfb0d1bf84fd279940508f7a834211fc58aecbbdaca502361ee46d6d30c`.
  QA checked 720 rows and a frozen 16-translation manual sample. Remaining
  flags were four Spanish and 92 Japanese number-format differences, one
  Spanish and 24 Japanese choice-label punctuation differences; no remaining
  detected empty/identical/extreme-length/preamble/answer-generation/target-
  script failures. Japanese semantic nuance was not human-fluency validated.
- Extracted both languages using pinned Qwen revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, `Qwen3_5ForCausalLM`, block
  hooks 1-32, final rightmost non-padding token, batch size 4, consecutive
  filtered frozen-manifest rows, dynamic right padding, no truncation, no
  generation, and float32 storage. Frozen dataset SHA-256 was rechecked
  immediately before each extraction. Total runner time was 149.103 seconds.
- Spanish extraction: 22.717 seconds, 15.847 prompts/s, peak allocated VRAM
  17,766.206 MiB, shape `(360, 32, 4096)`, artifact SHA-256
  `27a37b4bc363314fa06e0380a6190ea6bde0f7bdb10885ea0849fc2a86550585`.
  Japanese: 16.091 seconds, 22.372 prompts/s, peak allocated VRAM 17,760.370
  MiB, same shape, artifact SHA-256
  `4ff9680557bc903d42df91f870a0e462a40735dfc17d51ea179d8391ec9cb906`.
  Both passed finite-float32, metadata/order, exact save/reload checks.
- Applied frozen English Probe B block 32 artifact
  `c6039d72695ac36fe2c422b5d563e8f98ed90fb5b743c1e1a29f67fbc084652d`
  without fitting or refitting. AUROC / accuracy: EN-to-EN 0.997284 / 0.966667;
  EN-to-ES 0.881265 / 0.575000; EN-to-JA 0.864259 / 0.500000. Mean scores shifted
  from -0.193214 (EN) to -7.312260 (ES) and -11.162112 (JA); fixed-threshold
  predicted-evaluation rates were 0.527778, 0.075000, and 0.000000.
- Interpretation remains explicitly source-confounded, not clean evaluation
  awareness. The multilingual result preserves substantial rank information
  but has severe language-dependent score/calibration shifts.
- From the saved prediction CSV only (no transformer inference), paired target-
  minus-English block-32 score shifts, reported as mean / population SD over
  90 stable-ID-aligned examples per quadrant, were: EN-to-ES bench-eval
  -14.098305 / 5.933570, casual-eval -16.200841 / 3.625197, bench-deploy
  +1.307466 / 5.071057, and casual-deploy +0.515495 / 3.913767; EN-to-JA
  bench-eval -18.465254 / 3.938013, casual-eval -20.870874 / 3.606263,
  bench-deploy -2.207883 / 5.236618, and casual-deploy -2.331581 / 3.971470.

## 2026-09-05 - TF-IDF v2 selection-sensitivity diagnostic

- Reproduced the frozen-v2 training-only five-fold TF-IDF plus L2 logistic-
  regression mean CV AUROC exactly: 0.895692 (folds 0.920918, 0.894274,
  0.888464, 0.899660, 0.875142). The vectorizer and classifier were newly fit
  inside each training fold.
- Drew 20 deterministic balanced unmatched samples (seeds 42-61), each with
  210 rows per quadrant and 42 rows per quadrant per fold, from the exact-
  deduplicated/manual-exclusion source pools. Sampling was independent within
  quadrant, without pairwise matching, and restricted to the original v2
  inclusive common character-length supports: benchmark 103-672 and casual
  52-711. Frozen held-out IDs were excluded before sampling; held-out labels
  were not consulted.
- The 20 unmatched sample mean CV AUROCs averaged 0.913403 (population SD
  0.008427; range 0.898044-0.931604). Every seed mean exceeded frozen v2;
  frozen-v2 minus unmatched mean was -0.017711.
- Frozen-v2 format-specific training-only mean CV AUROC was 0.886168 for
  benchmark (fold range 0.857143-0.911565) and 0.963379 for casual (fold range
  0.948413-0.979025).
- Runtime was 10.585 seconds with Python 3.12.3, NumPy 1.26.4, and
  scikit-learn 1.5.2. Source dataset revision was
  `a50e4c983e7e66ebbe8f160e3549cc98e29271d4`; frozen-v2 SHA-256 remained
  `4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639`.
- Conclusion: v2 pairwise length matching did not create the high lexical
  identifiability. Unmatched length-support-controlled samples were slightly
  more identifiable, strengthening the source/provenance-confounding concern.
