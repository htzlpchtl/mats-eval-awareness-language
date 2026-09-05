"""Extract frozen English final-token block activations (Stage 5)."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.utils import final_nonpadding_indices, select_final_nonpadding


MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
FROZEN_ENGLISH_V2 = Path("data/selected/english_selected_v2.jsonl")
FROZEN_ENGLISH_V2_SHA256 = (
    "4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639"
)
QUADRANTS = ("bench_eval", "casual_eval", "bench_deploy", "casual_deploy")
EXPECTED_ROWS = 1200
ROWS_PER_QUADRANT = 300
N_LAYERS = 32
HIDDEN_SIZE = 4096
BATCH_SIZE = 4
PREFLIGHT_PER_QUADRANT = 10
REQUIRED_POLICY = {
    "batch_size": 4,
    "batch_order": "frozen_manifest_order",
    "batch_membership": "consecutive_rows",
    "padding": "dynamic",
    "padding_side": "right",
    "truncation": False,
    "final_token_selection": "rightmost_nonzero_attention_mask",
    "activation_mapping": "raw_outputs_all_32_decoder_blocks_via_forward_hooks",
    "storage_dtype": "float32",
    "languages": ["english", "spanish", "japanese"],
    "same_extraction_path_all_languages": True,
}
ARTIFACT_KEYS = {
    "activations",
    "row_indices",
    "stable_ids",
    "quadrants",
    "splits",
    "cv_folds",
    "context_labels",
    "format_labels",
    "pair_ids",
    "layer_numbers",
    "layer_module_indices",
    "layer_sources",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_mib(value: int) -> float:
    return value / (1024**2)


def json_default(value: Any) -> Any:
    """Serialize set-valued loading metadata and NumPy scalar values."""

    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def load_frozen_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Frozen v2 SHA mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    validate_frozen_rows(rows)
    return rows


def validate_frozen_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} frozen rows, found {len(rows)}")
    required_fields = {
        "stable_id",
        "quadrant",
        "split",
        "cv_fold",
        "context_label",
        "format_label",
        "pair_id",
        "model_facing_text",
    }
    for index, row in enumerate(rows):
        missing = required_fields - row.keys()
        if missing:
            raise RuntimeError(f"Frozen row {index} is missing fields: {sorted(missing)}")
        if not isinstance(row["model_facing_text"], str) or not row[
            "model_facing_text"
        ]:
            raise RuntimeError(f"Frozen row {index} has empty model-facing text")
    stable_ids = [row["stable_id"] for row in rows]
    if len(stable_ids) != len(set(stable_ids)):
        raise RuntimeError("Frozen stable IDs are not globally unique")
    counts = Counter((row["quadrant"], row["split"]) for row in rows)
    expected_counts = Counter(
        {(quadrant, "train"): 210 for quadrant in QUADRANTS}
        | {(quadrant, "test"): 90 for quadrant in QUADRANTS}
    )
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected frozen row counts: {counts}")
    contiguous_quadrant_segments(rows)


def contiguous_quadrant_segments(
    rows: list[dict[str, Any]],
) -> list[tuple[str, int, int]]:
    """Return contiguous [start, stop) quadrant segments in manifest order."""

    if not rows:
        raise RuntimeError("Cannot segment an empty frozen dataset")
    segments: list[tuple[str, int, int]] = []
    start = 0
    current = rows[0]["quadrant"]
    for index in range(1, len(rows)):
        quadrant = rows[index]["quadrant"]
        if quadrant != current:
            segments.append((current, start, index))
            current = quadrant
            start = index
    segments.append((current, start, len(rows)))
    segment_quadrants = [quadrant for quadrant, _, _ in segments]
    if len(segments) != len(QUADRANTS) or set(segment_quadrants) != set(QUADRANTS):
        raise RuntimeError(
            "Frozen rows must contain exactly one contiguous segment per quadrant; "
            f"found {segment_quadrants}"
        )
    if any(stop - start != ROWS_PER_QUADRANT for _, start, stop in segments):
        raise RuntimeError(f"Every quadrant segment must have {ROWS_PER_QUADRANT} rows")
    return segments


def choose_preflight_indices(rows: list[dict[str, Any]]) -> list[int]:
    """Choose ten manifest-order positions spanning every quadrant block."""

    selected: list[int] = []
    for _quadrant, start, stop in contiguous_quadrant_segments(rows):
        relative = np.rint(
            np.linspace(0, stop - start - 1, PREFLIGHT_PER_QUADRANT)
        ).astype(int)
        selected.extend((start + relative).tolist())
    selected.sort()
    if len(selected) != 40 or len(set(selected)) != 40:
        raise RuntimeError("Stage 5 preflight selection must contain 40 unique rows")
    counts = Counter(rows[index]["quadrant"] for index in selected)
    if counts != Counter({quadrant: 10 for quadrant in QUADRANTS}):
        raise RuntimeError(f"Unexpected preflight quadrant counts: {counts}")
    return selected


def render_and_measure(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[list[str], list[int]]:
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["model_facing_text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    lengths = [len(token_ids) for token_ids in tokenized]
    if len(lengths) != len(rows) or any(length <= 0 for length in lengths):
        raise RuntimeError("Token-length preflight did not return one nonempty input per row")
    return rendered, lengths


def token_length_summary(lengths: list[int], checked_limit: int) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(values.max()),
        "checked_model_limit": int(checked_limit),
        "number_exceeding_checked_model_limit": int(np.sum(values > checked_limit)),
        "truncation": False,
    }


def extraction_metadata(
    rows: list[dict[str, Any]], row_indices: np.ndarray
) -> dict[str, np.ndarray]:
    if len(rows) != len(row_indices):
        raise ValueError("Metadata rows and row indices must have the same length")
    return {
        "row_indices": np.asarray(row_indices, dtype=np.int32),
        "stable_ids": np.asarray([row["stable_id"] for row in rows]),
        "quadrants": np.asarray([row["quadrant"] for row in rows]),
        "splits": np.asarray([row["split"] for row in rows]),
        "cv_folds": np.asarray(
            [-1 if row["cv_fold"] is None else row["cv_fold"] for row in rows],
            dtype=np.int16,
        ),
        "context_labels": np.asarray(
            [row["context_label"] for row in rows], dtype=np.int8
        ),
        "format_labels": np.asarray(
            [row["format_label"] for row in rows], dtype=np.int8
        ),
        "pair_ids": np.asarray([row["pair_id"] for row in rows]),
        "layer_numbers": np.arange(1, N_LAYERS + 1, dtype=np.int16),
        "layer_module_indices": np.arange(N_LAYERS, dtype=np.int16),
        "layer_sources": np.asarray(
            [
                f"raw hook output of model.model.layers[{index}]"
                for index in range(N_LAYERS)
            ]
        ),
    }


def save_activation_artifact(
    path: Path,
    rows: list[dict[str, Any]],
    row_indices: np.ndarray,
    activations: np.ndarray,
    *,
    hidden_size: int = HIDDEN_SIZE,
) -> None:
    expected_shape = (len(rows), N_LAYERS, hidden_size)
    if activations.shape != expected_shape:
        raise RuntimeError(
            f"Activation shape must be {expected_shape}, got {activations.shape}"
        )
    if activations.dtype != np.float32:
        raise RuntimeError(f"Activations must be float32, got {activations.dtype}")
    if not np.isfinite(activations).all():
        raise RuntimeError("Activations contain NaN or infinity")
    arrays = extraction_metadata(rows, row_indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, activations=activations, **arrays)


def verify_activation_artifact(
    path: Path,
    rows: list[dict[str, Any]],
    row_indices: np.ndarray,
    expected_activations: np.ndarray | None = None,
    *,
    hidden_size: int = HIDDEN_SIZE,
) -> dict[str, Any]:
    expected_metadata = extraction_metadata(rows, row_indices)
    expected_shape = (len(rows), N_LAYERS, hidden_size)
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != ARTIFACT_KEYS:
            raise RuntimeError(
                f"Artifact keys differ: expected {sorted(ARTIFACT_KEYS)}, "
                f"got {sorted(artifact.files)}"
            )
        for key, expected in expected_metadata.items():
            if not np.array_equal(artifact[key], expected):
                raise RuntimeError(f"Artifact metadata mismatch for {key}")
        stored = artifact["activations"]
        if stored.shape != expected_shape:
            raise RuntimeError(
                f"Stored activation shape must be {expected_shape}, got {stored.shape}"
            )
        if stored.dtype != np.float32:
            raise RuntimeError(f"Stored activations must be float32, got {stored.dtype}")
        if not np.isfinite(stored).all():
            raise RuntimeError("Stored activations contain NaN or infinity")
        values_exact = (
            None
            if expected_activations is None
            else bool(np.array_equal(stored, expected_activations))
        )
        if expected_activations is not None and not values_exact:
            raise RuntimeError("Stored activation values changed after save/reload")
    return {
        "shape": list(expected_shape),
        "dtype": "float32",
        "metadata_exact": True,
        "values_exact": values_exact,
        "all_finite": True,
    }


def extract_indices(
    model: Any,
    tokenizer: Any,
    rendered: list[str],
    token_lengths: list[int],
    source_indices: list[int],
    *,
    batch_size: int,
    require_consecutive: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if require_consecutive and any(
        right != left + 1 for left, right in zip(source_indices, source_indices[1:])
    ):
        raise RuntimeError("Full extraction indices are not consecutive manifest rows")
    device = next(model.parameters()).device
    blocks = list(model.model.layers)
    if len(blocks) != N_LAYERS:
        raise RuntimeError(f"Loaded model has {len(blocks)} decoder blocks")
    output = np.empty((len(source_indices), N_LAYERS, HIDDEN_SIZE), dtype=np.float32)
    captures: dict[int, Any] = {}

    def capture(layer_index: int):
        def hook(_module: Any, _inputs: Any, layer_output: Any) -> None:
            captures[layer_index] = (
                layer_output[0] if isinstance(layer_output, tuple) else layer_output
            )

        return hook

    hooks = [
        block.register_forward_hook(capture(index))
        for index, block in enumerate(blocks)
    ]
    total_nonpadding_tokens = 0
    total_padded_tokens = 0
    batch_seconds: list[float] = []
    started = time.perf_counter()
    try:
        for output_start in range(0, len(source_indices), batch_size):
            batch_indices = source_indices[output_start : output_start + batch_size]
            if len(batch_indices) != batch_size:
                raise RuntimeError(
                    f"Extraction batch had {len(batch_indices)} rows, expected {batch_size}"
                )
            cpu_batch = tokenizer(
                [rendered[index] for index in batch_indices],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            observed_lengths = cpu_batch["attention_mask"].sum(dim=1).tolist()
            expected_lengths = [token_lengths[index] for index in batch_indices]
            if observed_lengths != expected_lengths:
                raise RuntimeError(
                    f"Batch token lengths changed for rows {batch_indices}: "
                    f"expected {expected_lengths}, got {observed_lengths}"
                )
            final_positions = final_nonpadding_indices(cpu_batch["attention_mask"])
            if not torch.equal(
                final_positions, cpu_batch["attention_mask"].sum(dim=1) - 1
            ):
                raise RuntimeError("Right-padding final-token invariant failed")
            total_nonpadding_tokens += int(cpu_batch["attention_mask"].sum().item())
            total_padded_tokens += int(cpu_batch["attention_mask"].numel())
            cuda_batch = {key: value.to(device) for key, value in cpu_batch.items()}
            captures.clear()
            torch.cuda.synchronize()
            batch_started = time.perf_counter()
            with torch.inference_mode():
                model_output = model(
                    **cuda_batch,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
                returned_states = tuple(model_output.hidden_states or ())
                if len(returned_states) != N_LAYERS + 1:
                    raise RuntimeError(
                        f"Forward returned {len(returned_states)} hidden states, expected 33"
                    )
                if set(captures) != set(range(N_LAYERS)):
                    raise RuntimeError(
                        f"Forward hooks captured layers {sorted(captures)}, expected 0-31"
                    )
                batch_activations = torch.stack(
                    [
                        select_final_nonpadding(
                            captures[index], cuda_batch["attention_mask"]
                        )
                        for index in range(N_LAYERS)
                    ],
                    dim=1,
                )
                if tuple(batch_activations.shape) != (
                    len(batch_indices),
                    N_LAYERS,
                    HIDDEN_SIZE,
                ):
                    raise RuntimeError(
                        f"Unexpected batch activation shape: {batch_activations.shape}"
                    )
                if not bool(torch.isfinite(batch_activations).all().item()):
                    raise RuntimeError("Batch activations contain NaN or infinity")
                cpu_activations = batch_activations.float().cpu().numpy()
            torch.cuda.synchronize()
            batch_seconds.append(time.perf_counter() - batch_started)
            output_stop = output_start + len(batch_indices)
            output[output_start:output_stop] = cpu_activations
            del (
                model_output,
                returned_states,
                batch_activations,
                cpu_activations,
                cuda_batch,
                cpu_batch,
            )
            captures.clear()
    finally:
        for hook in hooks:
            hook.remove()
    elapsed = time.perf_counter() - started
    if not np.isfinite(output).all():
        raise RuntimeError("Extracted activations contain NaN or infinity")
    return output, {
        "rows": len(source_indices),
        "batches": len(batch_seconds),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "batch_forward_and_transfer_seconds": batch_seconds,
        "prompts_per_second": len(source_indices) / elapsed,
        "nonpadding_input_tokens": total_nonpadding_tokens,
        "padded_input_tokens": total_padded_tokens,
        "padding_fraction": 1.0 - (total_nonpadding_tokens / total_padded_tokens),
        "all_finite": True,
    }


def checkpoint_path(directory: Path, segment_index: int, quadrant: str) -> Path:
    return directory / f"stage5_english_{segment_index:02d}_{quadrant}.npz"


def ensure_outputs_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to overwrite existing Stage 5 outputs: " + ", ".join(existing)
        )


def main() -> None:
    import accelerate
    import datasets
    import huggingface_hub
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--dataset", type=Path, default=FROZEN_ENGLISH_V2)
    parser.add_argument("--dataset-sha256", default=FROZEN_ENGLISH_V2_SHA256)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/activations/stage5_english_v2.npz"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/activations"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/metrics/stage5_english_manifest.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/metrics/stage5_summary.json"),
    )
    args = parser.parse_args()

    program_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if args.dataset.resolve() != FROZEN_ENGLISH_V2.resolve():
        raise RuntimeError(f"Stage 5 is restricted to {FROZEN_ENGLISH_V2}")
    if args.dataset_sha256 != FROZEN_ENGLISH_V2_SHA256:
        raise RuntimeError("Stage 5 requires the pinned frozen-v2 SHA-256")
    if args.model != MODEL_ID or args.revision != MODEL_REVISION:
        raise RuntimeError("Stage 5 requires the pinned Qwen model and revision")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5 requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")

    project_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if project_config["activation_extraction"] != REQUIRED_POLICY:
        raise RuntimeError(
            "config.yaml does not contain the frozen Stage 4 extraction policy"
        )
    rows = load_frozen_rows(args.dataset, args.dataset_sha256)
    segments = contiguous_quadrant_segments(rows)
    checkpoint_paths = [
        checkpoint_path(args.checkpoint_directory, index, quadrant)
        for index, (quadrant, _, _) in enumerate(segments, start=1)
    ]
    ensure_outputs_absent(
        checkpoint_paths + [args.artifact, args.manifest, args.report]
    )

    hub_info = HfApi().model_info(args.model, revision=args.revision)
    if hub_info.sha != args.revision:
        raise RuntimeError(
            f"Pinned revision resolved to {hub_info.sha}, expected {args.revision}"
        )
    top_config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    if not hasattr(top_config, "text_config"):
        raise RuntimeError("Qwen3.5 top-level config has no text_config")
    text_config = top_config.text_config
    if (
        text_config.num_hidden_layers != N_LAYERS
        or text_config.hidden_size != HIDDEN_SIZE
    ):
        raise RuntimeError(
            "Unexpected Qwen language architecture: "
            f"{text_config.num_hidden_layers} layers, hidden size "
            f"{text_config.hidden_size}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rendered, token_lengths = render_and_measure(tokenizer, rows)
    model_limits = [int(text_config.max_position_embeddings)]
    if 0 < int(tokenizer.model_max_length) < 1_000_000_000:
        model_limits.append(int(tokenizer.model_max_length))
    checked_limit = min(model_limits)
    length_summary = token_length_summary(token_lengths, checked_limit)
    if length_summary["number_exceeding_checked_model_limit"]:
        raise RuntimeError(
            "One or more prompts exceeds the checked model input limit; "
            "truncation is not authorised"
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_load_started = time.perf_counter()
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        config=text_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        output_loading_info=True,
    )
    model.eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    missing_keys = loading_info.get("missing_keys", [])
    mismatched_keys = loading_info.get("mismatched_keys", [])
    error_messages = loading_info.get("error_msgs", [])
    unexpected_keys = loading_info.get("unexpected_keys", [])
    if missing_keys or mismatched_keys or error_messages:
        raise RuntimeError(
            "Incomplete text-only checkpoint load: "
            f"missing={len(missing_keys)}, mismatched={len(mismatched_keys)}, "
            f"errors={error_messages}"
        )
    disallowed_unexpected = [
        key
        for key in unexpected_keys
        if not (key.startswith("model.visual.") or key.startswith("mtp."))
    ]
    if disallowed_unexpected:
        raise RuntimeError(
            f"Unexpected non-vision/non-MTP weights: {disallowed_unexpected[:10]}"
        )
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise RuntimeError(f"Unexpected loaded model class: {type(model).__name__}")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    parameter_devices = sorted(
        {str(parameter.device) for parameter in model.parameters()}
    )
    if parameter_dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"Unexpected parameter dtypes: {parameter_dtypes}")
    model_load_memory = {
        "allocated_mib": to_mib(torch.cuda.memory_allocated()),
        "reserved_mib": to_mib(torch.cuda.memory_reserved()),
        "peak_allocated_mib": to_mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": to_mib(torch.cuda.max_memory_reserved()),
    }

    preflight_dataset_sha256 = sha256_file(args.dataset)
    if preflight_dataset_sha256 != args.dataset_sha256:
        raise RuntimeError("Frozen v2 changed immediately before the preflight")
    preflight_indices = choose_preflight_indices(rows)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    preflight_activations, preflight_stats = extract_indices(
        model,
        tokenizer,
        rendered,
        token_lengths,
        preflight_indices,
        batch_size=BATCH_SIZE,
        require_consecutive=False,
    )
    if preflight_activations.shape != (40, N_LAYERS, HIDDEN_SIZE):
        raise RuntimeError(
            f"Unexpected preflight shape: {preflight_activations.shape}"
        )
    preflight_stats.update(
        {
            "selection": "10 evenly spaced manifest positions per quadrant",
            "quadrant_counts": dict(
                Counter(rows[index]["quadrant"] for index in preflight_indices)
            ),
            "source_row_indices": preflight_indices,
            "activation_shape": list(preflight_activations.shape),
            "peak_allocated_mib": to_mib(torch.cuda.max_memory_allocated()),
            "peak_reserved_mib": to_mib(torch.cuda.max_memory_reserved()),
            "dataset_sha256_immediately_before": preflight_dataset_sha256,
            "assertions_passed": True,
        }
    )
    del preflight_activations

    full_dataset_sha256 = sha256_file(args.dataset)
    if full_dataset_sha256 != args.dataset_sha256:
        raise RuntimeError("Frozen v2 changed immediately before full extraction")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    full_run_started = time.perf_counter()
    quadrant_reports: list[dict[str, Any]] = []
    extraction_seconds = 0.0
    for segment_index, ((quadrant, start, stop), path) in enumerate(
        zip(segments, checkpoint_paths, strict=True), start=1
    ):
        source_indices = list(range(start, stop))
        segment_rows = rows[start:stop]
        activations, extraction_stats = extract_indices(
            model,
            tokenizer,
            rendered,
            token_lengths,
            source_indices,
            batch_size=BATCH_SIZE,
            require_consecutive=True,
        )
        extraction_seconds += extraction_stats["elapsed_seconds"]
        checkpoint_save_started = time.perf_counter()
        save_activation_artifact(
            path,
            segment_rows,
            np.asarray(source_indices),
            activations,
        )
        checkpoint_save_seconds = time.perf_counter() - checkpoint_save_started
        reload_checks = verify_activation_artifact(
            path,
            segment_rows,
            np.asarray(source_indices),
            activations,
        )
        quadrant_reports.append(
            {
                "segment_index": segment_index,
                "quadrant": quadrant,
                "source_row_start": start,
                "source_row_stop_exclusive": stop,
                **extraction_stats,
                "checkpoint_path": str(path),
                "checkpoint_sha256": sha256_file(path),
                "checkpoint_size_bytes": path.stat().st_size,
                "checkpoint_save_seconds": checkpoint_save_seconds,
                "checkpoint_reload": reload_checks,
            }
        )
        del activations
    full_run_seconds = time.perf_counter() - full_run_started
    full_peak_allocated_mib = to_mib(torch.cuda.max_memory_allocated())
    full_peak_reserved_mib = to_mib(torch.cuda.max_memory_reserved())

    assemble_started = time.perf_counter()
    all_activations = np.empty(
        (EXPECTED_ROWS, N_LAYERS, HIDDEN_SIZE), dtype=np.float32
    )
    seen = np.zeros(EXPECTED_ROWS, dtype=bool)
    for (_quadrant, start, stop), path in zip(
        segments, checkpoint_paths, strict=True
    ):
        with np.load(path, allow_pickle=False) as checkpoint:
            row_indices = checkpoint["row_indices"]
            expected_indices = np.arange(start, stop, dtype=np.int32)
            if not np.array_equal(row_indices, expected_indices):
                raise RuntimeError(f"Checkpoint row indices are misaligned: {path}")
            if seen[row_indices].any():
                raise RuntimeError(f"Checkpoint row indices overlap: {path}")
            all_activations[row_indices] = checkpoint["activations"]
            seen[row_indices] = True
    if not seen.all():
        raise RuntimeError("Checkpoint assembly did not cover all frozen rows")
    save_activation_artifact(
        args.artifact,
        rows,
        np.arange(EXPECTED_ROWS, dtype=np.int32),
        all_activations,
    )
    final_reload_checks = verify_activation_artifact(
        args.artifact,
        rows,
        np.arange(EXPECTED_ROWS, dtype=np.int32),
        all_activations,
    )
    artifact_assembly_save_reload_seconds = time.perf_counter() - assemble_started
    artifact_sha256 = sha256_file(args.artifact)
    dataset_sha256_after = sha256_file(args.dataset)
    if dataset_sha256_after != args.dataset_sha256:
        raise RuntimeError("Frozen v2 changed during Stage 5")

    layer_mapping = [
        {
            "stored_index": index,
            "transformer_block_1_based": index + 1,
            "module": f"model.model.layers[{index}]",
            "source": "raw forward-hook output at final rightmost non-padding token",
        }
        for index in range(N_LAYERS)
    ]
    manifest = {
        "stage": 5,
        "language": "english",
        "artifact": {
            "path": str(args.artifact),
            "sha256": artifact_sha256,
            "shape": list(all_activations.shape),
            "dtype": "float32",
            "row_axis": "frozen JSONL manifest order",
            "layer_axis": "transformer blocks 1-32",
            "hidden_axis": "Qwen text hidden dimension",
        },
        "dataset": {
            "path": str(args.dataset),
            "sha256": dataset_sha256_after,
        },
        "layer_mapping": layer_mapping,
        "rows": [
            {
                "activation_row": index,
                "stable_id": row["stable_id"],
                "quadrant": row["quadrant"],
                "split": row["split"],
                "cv_fold": row["cv_fold"],
                "pair_id": row["pair_id"],
            }
            for index, row in enumerate(rows)
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, default=json_default) + "\n",
        encoding="utf-8",
    )

    extraction_totals = {
        "rows": sum(report["rows"] for report in quadrant_reports),
        "batches": sum(report["batches"] for report in quadrant_reports),
        "nonpadding_input_tokens": sum(
            report["nonpadding_input_tokens"] for report in quadrant_reports
        ),
        "padded_input_tokens": sum(
            report["padded_input_tokens"] for report in quadrant_reports
        ),
        "extraction_seconds_excluding_checkpoint_saves": extraction_seconds,
        "prompts_per_second_excluding_checkpoint_saves": EXPECTED_ROWS
        / extraction_seconds,
        "full_run_seconds_including_checkpoint_saves_and_reloads": full_run_seconds,
        "prompts_per_second_including_checkpoint_saves_and_reloads": EXPECTED_ROWS
        / full_run_seconds,
        "peak_allocated_mib": full_peak_allocated_mib,
        "peak_reserved_mib": full_peak_reserved_mib,
    }
    report = {
        "stage": 5,
        "checkpoint": 5,
        "started_at_utc": started_at,
        "seed": 42,
        "language": "english",
        "dataset": {
            "path": str(args.dataset),
            "expected_sha256": args.dataset_sha256,
            "sha256_immediately_before_preflight": preflight_dataset_sha256,
            "sha256_immediately_before_full_extraction": full_dataset_sha256,
            "sha256_after_extraction": dataset_sha256_after,
            "rows": len(rows),
            "modified": False,
        },
        "model": {
            "id": args.model,
            "requested_revision": args.revision,
            "resolved_revision": hub_info.sha,
            "loaded_class": type(model).__name__,
            "language_model_blocks": len(model.model.layers),
            "hidden_size": model.config.hidden_size,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "parameter_dtypes": parameter_dtypes,
            "parameter_devices": parameter_devices,
            "loading_info": {
                "missing_keys": missing_keys,
                "mismatched_keys": mismatched_keys,
                "unexpected_key_count": len(unexpected_keys),
                "error_messages": error_messages,
            },
        },
        "policy": REQUIRED_POLICY,
        "forward": {
            "inference_mode": True,
            "generated_answers": False,
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
            "logits_to_keep": 1,
            "chat_messages": "one user message and no system message",
            "add_generation_prompt": True,
        },
        "token_lengths": length_summary,
        "preflight_40": preflight_stats,
        "full_extraction": extraction_totals,
        "quadrants": quadrant_reports,
        "layer_mapping": layer_mapping,
        "artifact": {
            "path": str(args.artifact),
            "sha256": artifact_sha256,
            "size_bytes": args.artifact.stat().st_size,
            "shape": list(all_activations.shape),
            "dtype": "float32",
            "reload_checks": final_reload_checks,
            "manifest_path": str(args.manifest),
        },
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "full_extraction_seconds_excluding_checkpoint_saves": extraction_seconds,
            "full_run_seconds_including_quadrant_checkpoints": full_run_seconds,
            "artifact_assembly_save_reload_seconds": artifact_assembly_save_reload_seconds,
            "program_to_final_artifact_verification_seconds": time.perf_counter()
            - program_started,
        },
        "gpu_memory": {
            "after_model_load": model_load_memory,
            "full_extraction_peak_allocated_mib": full_peak_allocated_mib,
            "full_extraction_peak_reserved_mib": full_peak_reserved_mib,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "checks": {
            "preflight_passed": True,
            "exactly_1200_rows": len(rows) == EXPECTED_ROWS,
            "activation_shape_exact": all_activations.shape
            == (EXPECTED_ROWS, N_LAYERS, HIDDEN_SIZE),
            "all_values_finite": bool(np.isfinite(all_activations).all()),
            "stable_ids_and_order_exact": final_reload_checks["metadata_exact"],
            "quadrants_and_splits_exact": final_reload_checks["metadata_exact"],
            "layer_numbering_explicit_and_correct": True,
            "save_reload_values_exact": final_reload_checks["values_exact"],
            "dataset_unchanged": True,
            "stage6_started": False,
        },
        "warnings": [
            "Hugging Face Hub access was unauthenticated.",
            *(
                [
                    f"Allowed vision/MTP-only unexpected checkpoint keys: "
                    f"{len(unexpected_keys)}"
                ]
                if unexpected_keys
                else []
            ),
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
