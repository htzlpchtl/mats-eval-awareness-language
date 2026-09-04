# Compute setup for Qwen3.5-9B

This is the Checkpoint 3 handoff. It prepares a remote GPU machine for the
Stage 4 smoke test; it does not itself run that smoke test or download model
weights.

## Required machine

- One NVIDIA A40 48 GB or RTX A6000 48 GB.
- At least 80 GB of usable persistent storage.
- A recent Linux NVIDIA driver and SSH access.
- Outbound access to GitHub, PyPI, and Hugging Face.

Qwen3.5-9B uses BF16 weights. The official model card lists 9 billion
parameters, 32 language-model layers, and hidden size 4096. The expected BF16
weight footprint is about 18 GB before runtime overhead, so a 48 GB card is the
initial target.

Official references:

- [Qwen/Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3.5 configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Transformers installation guide](https://huggingface.co/docs/transformers/installation)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)

## 1. Inspect the machine before installing anything

Run these commands after connecting over SSH:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
df -h .
python3 --version
git --version
```

Continue only if the selected GPU is an A40 48 GB or RTX A6000 48 GB, roughly
80 GB or more storage is free, and Python is version 3.10 or newer. Python 3.11
is preferred for this run.

## 2. Clone or update the repository

For a fresh machine:

```bash
export REPO_URL="https://github.com/htzlpchtl/mats-eval-awareness-language.git"
git clone "$REPO_URL"
cd mats-eval-awareness-language
git switch main
git pull --ff-only origin main
```

For an existing checkout:

```bash
cd mats-eval-awareness-language
git fetch origin
git switch main
git pull --ff-only origin main
```

Record and inspect the exact code revision:

```bash
git rev-parse HEAD
git status --short --branch
```

The status must be clean before the smoke test. Record the printed commit in
`RUN_LOG.md` when Stage 4 begins.

Verify that the approved v2 dataset is present and byte-identical:

```bash
export V2_DATA="data/selected/english_selected_v2.jsonl"
export V2_SHA256="4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639"
test "$(sha256sum "$V2_DATA" | cut -d ' ' -f 1)" = "$V2_SHA256"
echo "Verified v2 dataset SHA-256: $V2_SHA256"
```

The v1 files are a retired pilot. All subsequent model work must use
`data/selected/english_selected_v2.jsonl`.

## 3. Create the Python environment

Use Python 3.11 when it is available:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade -r requirements.txt
```

If the image provides only a supported `python3` (3.10 or newer), replace
`python3.11` in the first command with `python3`.

The project requirement `transformers>=4.57,<6` covers the Qwen3.5 model type
recorded by the official configuration. The compatibility check below also
requires the official auto classes before any weights are fetched. Do not
silently fall back to a custom model implementation.

## 4. Configure persistent caches

Keep downloaded model files on the machine's persistent volume. From the
repository root, the following uses the ignored local `.cache` directory:

```bash
export PROJECT_DIR="$PWD"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"
```

Re-export these variables after reconnecting. If the repository is not on the
persistent volume, set `HF_HOME` to a directory on that volume instead.

## 5. Optional Hugging Face authentication

`Qwen/Qwen3.5-9B` is public, so authentication is normally unnecessary. If the
host encounters rate limits, authenticate interactively without placing a
token in shell history, source files, or this repository:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN
hf auth login --token "$HF_TOKEN"
unset HF_TOKEN
```

## 6. Verify packages and CUDA without downloading Qwen

```bash
python -c 'import sys, torch, transformers; print("python", sys.version); print("torch", torch.__version__); print("transformers", transformers.__version__); print("torch CUDA runtime", torch.version.cuda); print("CUDA available", torch.cuda.is_available()); assert torch.cuda.is_available(); print("GPU", torch.cuda.get_device_name(0)); print("GPU bytes", torch.cuda.get_device_properties(0).total_memory); print("BF16 supported", torch.cuda.is_bf16_supported()); assert torch.cuda.is_bf16_supported()'
python -c 'from transformers import AutoProcessor, AutoModelForMultimodalLM; print("Official Qwen3.5 auto classes available")'
python -m pytest -q -W error
python -m pip freeze
```

If `torch.cuda.is_available()` is false, stop. Confirm that the machine exposes
the GPU and use the official PyTorch selector to install a wheel compatible
with the host's NVIDIA driver; do not begin Stage 4 with a CPU-only build.

Before reporting the machine ready, rerun:

```bash
nvidia-smi
df -h "$PROJECT_DIR"
git status --short --branch
```

## 7. Pinned model identity for Stage 4

Use these values for the initial reproducible smoke test:

```bash
export MODEL_ID="Qwen/Qwen3.5-9B"
export MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
```

That model revision was returned by the Hugging Face Hub API on 2026-09-04.
Stage 4 should use the official model-card loading path:
`AutoProcessor.from_pretrained(...)` and
`AutoModelForMultimodalLM.from_pretrained(...)`, passing the pinned revision to
both calls. Do not run those calls during this setup checkpoint because they
will download the model.

## Handoff back to Codex

Send the following outputs after the machine is provisioned:

- `nvidia-smi` and the compact GPU query;
- `df -h "$PROJECT_DIR"`;
- `python3 --version` (and `python --version` inside the venv);
- the CUDA verification output;
- `git rev-parse HEAD` and `git status --short --branch`;
- confirmation that the v2 SHA-256 check passed.

Do not run extraction or inspect the final v2 test partition. Codex will begin
Stage 4 with a model/processor/hidden-state smoke test and stop at its next
checkpoint.
