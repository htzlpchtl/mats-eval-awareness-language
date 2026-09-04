"""Inspect Qwen architecture, chat templates, and hidden states (Stage 4)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.utils import final_nonpadding_indices, select_final_nonpadding


MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
FROZEN_V2 = Path("data/selected/english_selected_v2.jsonl")
FROZEN_V2_SHA256 = "4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639"
QUADRANTS = ("bench_eval", "casual_eval", "bench_deploy", "casual_deploy")
SMOKE_QUANTILES = {
    "bench_eval": 0.0,
    "casual_eval": 1.0 / 3.0,
    "bench_deploy": 2.0 / 3.0,
    "casual_deploy": 1.0,
}
BENCHMARK_QUANTILES = (0.2, 0.4, 0.6, 0.8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_v2(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Frozen v2 SHA mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != 1200:
        raise RuntimeError(f"Expected 1,200 frozen rows, found {len(rows)}")
    counts = Counter((row["quadrant"], row["split"]) for row in rows)
    expected = Counter(
        {(quadrant, "train"): 210 for quadrant in QUADRANTS}
        | {(quadrant, "test"): 90 for quadrant in QUADRANTS}
    )
    if counts != expected:
        raise RuntimeError(f"Unexpected frozen-v2 counts: {counts}")
    return rows


def sorted_training_rows(
    rows: list[dict[str, Any]], quadrant: str
) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if row["quadrant"] == quadrant and row["split"] == "train"
        ),
        key=lambda row: (row["character_count"], row["stable_id"]),
    )


def choose_smoke_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for quadrant in QUADRANTS:
        candidates = sorted_training_rows(rows, quadrant)
        quantile = SMOKE_QUANTILES[quadrant]
        selected.append(candidates[round(quantile * (len(candidates) - 1))])
    if {row["quadrant"] for row in selected} != set(QUADRANTS):
        raise RuntimeError("Smoke batch must contain one row from every quadrant")
    if any(row["split"] != "train" for row in selected):
        raise RuntimeError("Stage 4 smoke rows must all be training rows")
    return selected


def choose_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for quantile in BENCHMARK_QUANTILES:
        for quadrant in QUADRANTS:
            candidates = sorted_training_rows(rows, quadrant)
            selected.append(candidates[round(quantile * (len(candidates) - 1))])
    if len(selected) != 16 or len({row["stable_id"] for row in selected}) != 16:
        raise RuntimeError("Throughput benchmark must contain 16 unique rows")
    if any(row["split"] != "train" for row in selected):
        raise RuntimeError("Throughput benchmark rows must all be training rows")
    if Counter(row["quadrant"] for row in selected) != Counter(
        {quadrant: 4 for quadrant in QUADRANTS}
    ):
        raise RuntimeError("Throughput benchmark must contain four rows per quadrant")
    return selected


def compare_tensors(left: Any, right: Any) -> dict[str, Any]:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        return {"shape_match": False, "exact": False}
    difference = (left.float() - right.float()).abs()
    return {
        "shape_match": True,
        "exact": bool(torch.equal(left, right)),
        "max_abs_difference": float(difference.max().item()),
    }


def to_mib(value: int) -> float:
    return value / (1024**2)


def json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def tokenized_chat_batch(tokenizer: Any, rows: list[dict[str, Any]]) -> Any:
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["model_facing_text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    return tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def benchmark_throughput(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    total_tokens = 0
    total_padded_tokens = 0
    batch_seconds = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    final_block_capture: dict[str, Any] = {}

    def capture_final_block(_module: Any, _inputs: Any, output: Any) -> None:
        final_block_capture["state"] = (
            output[0] if isinstance(output, tuple) else output
        )

    final_block_hook = model.model.layers[-1].register_forward_hook(
        capture_final_block
    )
    benchmark_start = time.perf_counter()

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        cpu_batch = tokenized_chat_batch(tokenizer, batch_rows)
        total_tokens += int(cpu_batch["attention_mask"].sum().item())
        total_padded_tokens += int(cpu_batch["attention_mask"].numel())
        cuda_batch = {key: value.to(device) for key, value in cpu_batch.items()}
        torch.cuda.synchronize()
        batch_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(
                **cuda_batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            returned_states = tuple(outputs.hidden_states or ())
            if len(returned_states) != 33:
                raise RuntimeError(
                    f"Benchmark forward returned {len(returned_states)} hidden states"
                )
            block_states = returned_states[1:32] + (
                final_block_capture["state"],
            )
            activations = torch.stack(
                [
                    select_final_nonpadding(state, cuda_batch["attention_mask"])
                    for state in block_states
                ],
                dim=1,
            )
            if tuple(activations.shape) != (len(batch_rows), 32, 4096):
                raise RuntimeError(
                    f"Benchmark activation shape was {tuple(activations.shape)}"
                )
            if not bool(torch.isfinite(activations).all().item()):
                raise RuntimeError("Benchmark activations contain NaN or infinity")
            _ = activations.float().cpu().numpy()
        torch.cuda.synchronize()
        batch_seconds.append(time.perf_counter() - batch_start)
        del outputs, returned_states, block_states, activations, cuda_batch, cpu_batch

    final_block_hook.remove()
    elapsed = time.perf_counter() - benchmark_start
    prompts_per_second = len(rows) / elapsed
    tokens_per_second = total_tokens / elapsed
    projected_seconds = 1200 / prompts_per_second
    return {
        "selection": "four character-length quantiles per quadrant",
        "split": "train",
        "rows": len(rows),
        "quadrant_counts": dict(Counter(row["quadrant"] for row in rows)),
        "batch_size": batch_size,
        "batches": len(batch_seconds),
        "batch_seconds": batch_seconds,
        "elapsed_seconds": elapsed,
        "nonpadding_input_tokens": total_tokens,
        "padded_input_tokens": total_padded_tokens,
        "padding_fraction": 1.0 - (total_tokens / total_padded_tokens),
        "prompts_per_second": prompts_per_second,
        "nonpadding_tokens_per_second": tokens_per_second,
        "peak_allocated_mib": to_mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": to_mib(torch.cuda.max_memory_reserved()),
        "projected_1200_seconds": projected_seconds,
        "projected_1200_minutes": projected_seconds / 60.0,
        "includes_gpu_to_cpu_float32_transfer": True,
        "generated_answers": False,
    }


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
    parser.add_argument("--dataset", type=Path, default=FROZEN_V2)
    parser.add_argument("--dataset-sha256", default=FROZEN_V2_SHA256)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/activations/stage4_smoke.npz")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/metrics/stage4_smoke.json")
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4 requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")

    project_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    extraction_policy = project_config["activation_extraction"]
    required_policy = {
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
    if extraction_policy != required_policy:
        raise RuntimeError(f"Unexpected activation extraction policy: {extraction_policy}")

    rows = load_frozen_v2(args.dataset, args.dataset_sha256)
    smoke_rows = choose_smoke_rows(rows)
    benchmark_rows = choose_benchmark_rows(rows)

    hub_info = HfApi().model_info(args.model, revision=args.revision)
    if hub_info.sha != args.revision:
        raise RuntimeError(
            f"Pinned revision resolved to {hub_info.sha}, expected {args.revision}"
        )
    top_config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    if not hasattr(top_config, "text_config"):
        raise RuntimeError("Qwen3.5 top-level config has no text_config")
    text_config = top_config.text_config
    if text_config.num_hidden_layers != 32 or text_config.hidden_size != 4096:
        raise RuntimeError(
            "Unexpected language architecture: "
            f"{text_config.num_hidden_layers} layers, hidden size {text_config.hidden_size}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = extraction_policy["padding_side"]

    templates = []
    individual_ids = []
    prompt_reports = []
    for row in smoke_rows:
        messages = [{"role": "user", "content": row["model_facing_text"]}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenized_template = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        token_ids = tokenized_template["input_ids"]
        round_trip_ids = tokenizer(
            rendered, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if token_ids != round_trip_ids:
            raise RuntimeError("Rendered chat template did not round-trip")
        templates.append(rendered)
        individual_ids.append(token_ids)
        prompt_reports.append(
            {
                "stable_id": row["stable_id"],
                "quadrant": row["quadrant"],
                "split": row["split"],
                "selection_quantile": SMOKE_QUANTILES[row["quadrant"]],
                "model_facing_text": row["model_facing_text"],
                "templated_decoded_string": rendered,
                "token_count": len(token_ids),
                "final_10_token_ids": token_ids[-10:],
                "final_10_tokens": tokenizer.convert_ids_to_tokens(token_ids[-10:]),
            }
        )

    batch = tokenizer(
        templates,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    for index, expected_ids in enumerate(individual_ids):
        attended_ids = batch["input_ids"][index][
            batch["attention_mask"][index].bool()
        ].tolist()
        if attended_ids != expected_ids:
            raise RuntimeError(f"Dynamic padding changed smoke row {index}")

    final_positions = final_nonpadding_indices(batch["attention_mask"])
    sum_minus_one = batch["attention_mask"].sum(dim=1) - 1
    if not torch.equal(final_positions, sum_minus_one):
        raise RuntimeError("Right-padding final-token invariant failed")
    for index, prompt_report in enumerate(prompt_reports):
        position = int(final_positions[index].item())
        token_id = int(batch["input_ids"][index, position].item())
        prompt_report.update(
            {
                "padded_batch_length": int(batch["input_ids"].shape[1]),
                "padding_count": int(
                    batch["input_ids"].shape[1]
                    - batch["attention_mask"][index].sum().item()
                ),
                "final_nonpadding_index": position,
                "actual_final_token_id": token_id,
                "actual_final_token": tokenizer.convert_ids_to_tokens(token_id),
                "actual_final_token_decoded": tokenizer.decode([token_id]),
                "attention_mask": batch["attention_mask"][index].tolist(),
            }
        )

    if sha256_file(args.dataset) != args.dataset_sha256:
        raise RuntimeError("Frozen v2 changed between preflight and model load")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        config=text_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        output_loading_info=True,
    )
    model.eval()

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

    blocks = list(model.model.layers)
    if len(blocks) != 32:
        raise RuntimeError(f"Loaded model has {len(blocks)} language blocks")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"Unexpected parameter dtypes: {parameter_dtypes}")
    load_memory = {
        "allocated_mib": to_mib(torch.cuda.memory_allocated()),
        "reserved_mib": to_mib(torch.cuda.memory_reserved()),
        "peak_allocated_mib": to_mib(torch.cuda.max_memory_allocated()),
    }

    captures: dict[str, Any] = {}
    hooks = []

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captures[name] = output[0] if isinstance(output, tuple) else output

        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(capture("embeddings")))
    hooks.extend(
        block.register_forward_hook(capture(f"block_{index + 1}"))
        for index, block in enumerate(blocks)
    )
    hooks.append(model.model.norm.register_forward_hook(capture("final_norm")))

    device = next(model.parameters()).device
    cuda_batch = {key: value.to(device) for key, value in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        outputs = model(
            **cuda_batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()

    returned_states = tuple(outputs.hidden_states or ())
    if len(returned_states) != 33:
        raise RuntimeError(
            f"Expected embeddings plus 32 model states, got {len(returned_states)}"
        )
    returned_shapes = [list(state.shape) for state in returned_states]
    returned_dtype = str(returned_states[0].dtype)
    embedding_comparison = compare_tensors(
        returned_states[0], captures["embeddings"]
    )
    first_31_block_matches = [
        compare_tensors(returned_states[index], captures[f"block_{index}"])
        for index in range(1, 32)
    ]
    final_norm_comparison = compare_tensors(
        returned_states[32], captures["final_norm"]
    )
    returned_last_vs_block_32 = compare_tensors(
        returned_states[32], captures["block_32"]
    )
    if not embedding_comparison["exact"]:
        raise RuntimeError("Returned index 0 is not the embedding output")
    if not all(match["exact"] for match in first_31_block_matches):
        raise RuntimeError("Returned indices 1-31 do not match blocks 1-31")
    if not final_norm_comparison["exact"]:
        raise RuntimeError("Returned index 32 is not the terminal norm output")
    if returned_last_vs_block_32["exact"]:
        raise RuntimeError("Expected raw block 32 to differ from terminal norm")

    block_states = tuple(captures[f"block_{index}"] for index in range(1, 33))

    activations = torch.stack(
        [
            select_final_nonpadding(state, cuda_batch["attention_mask"])
            for state in block_states
        ],
        dim=1,
    )
    if tuple(activations.shape) != (4, 32, 4096):
        raise RuntimeError(f"Unexpected smoke activation shape: {activations.shape}")
    if not bool(torch.isfinite(activations).all().item()):
        raise RuntimeError("Smoke activations contain NaN or infinity")
    smoke_forward_memory = {
        "allocated_mib": to_mib(torch.cuda.memory_allocated()),
        "reserved_mib": to_mib(torch.cuda.memory_reserved()),
        "peak_allocated_mib": to_mib(torch.cuda.max_memory_allocated()),
    }

    single_batch = tokenizer(
        templates[0], add_special_tokens=False, return_tensors="pt"
    )
    single_cuda = {key: value.to(device) for key, value in single_batch.items()}
    single_block_32: dict[str, Any] = {}

    def capture_single_block_32(_module: Any, _inputs: Any, output: Any) -> None:
        single_block_32["state"] = (
            output[0] if isinstance(output, tuple) else output
        )

    single_hook = model.model.layers[-1].register_forward_hook(
        capture_single_block_32
    )
    with torch.inference_mode():
        single_outputs = model(
            **single_cuda,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    single_hook.remove()
    single_states = tuple(single_outputs.hidden_states or ())
    if len(single_states) != 33:
        raise RuntimeError(
            f"Single-example forward returned {len(single_states)} hidden states"
        )
    single_block_states = single_states[1:32] + (
        single_block_32["state"],
    )
    single_activations = torch.stack(
        [
            select_final_nonpadding(state, single_cuda["attention_mask"])
            for state in single_block_states
        ],
        dim=1,
    )[0]
    batched_first = activations[0]
    absolute_difference = (batched_first.float() - single_activations.float()).abs()
    max_abs_by_layer = absolute_difference.max(dim=1).values
    mean_abs_by_layer = absolute_difference.mean(dim=1)
    cosine_by_layer = torch.nn.functional.cosine_similarity(
        batched_first.float(), single_activations.float(), dim=1
    )
    relative_l2_by_layer = torch.linalg.vector_norm(
        batched_first.float() - single_activations.float(), dim=1
    ) / torch.maximum(
        torch.linalg.vector_norm(batched_first.float(), dim=1),
        torch.linalg.vector_norm(single_activations.float(), dim=1),
    )
    elementwise_allclose = bool(
        torch.allclose(
            batched_first.float(),
            single_activations.float(),
            rtol=2e-2,
            atol=2e-2,
        )
    )
    minimum_cosine_required = 0.999
    maximum_relative_l2_allowed = 0.05
    batch_single_agreement = bool(
        cosine_by_layer.min().item() >= minimum_cosine_required
        and relative_l2_by_layer.max().item() <= maximum_relative_l2_allowed
    )
    if not batch_single_agreement:
        raise RuntimeError(
            "Batched/single extraction mismatch: "
            f"minimum cosine={cosine_by_layer.min().item()}, "
            f"maximum relative L2={relative_l2_by_layer.max().item()}"
        )

    cpu_activations = activations.float().cpu().numpy()
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact,
        stable_ids=np.asarray([row["stable_id"] for row in smoke_rows]),
        quadrants=np.asarray([row["quadrant"] for row in smoke_rows]),
        layer_numbers=np.arange(1, 33, dtype=np.int16),
        activations=cpu_activations,
    )
    with np.load(args.artifact, allow_pickle=False) as reloaded:
        artifact_checks = {
            "ids_unchanged": bool(
                np.array_equal(
                    reloaded["stable_ids"],
                    np.asarray([row["stable_id"] for row in smoke_rows]),
                )
            ),
            "quadrants_unchanged": bool(
                np.array_equal(
                    reloaded["quadrants"],
                    np.asarray([row["quadrant"] for row in smoke_rows]),
                )
            ),
            "shape": list(reloaded["activations"].shape),
            "shape_unchanged": reloaded["activations"].shape
            == cpu_activations.shape,
            "values_exact": bool(np.array_equal(reloaded["activations"], cpu_activations)),
            "all_finite": bool(np.isfinite(reloaded["activations"]).all()),
        }
    required_artifact_checks = (
        "ids_unchanged",
        "quadrants_unchanged",
        "shape_unchanged",
        "values_exact",
        "all_finite",
    )
    if not all(artifact_checks[key] for key in required_artifact_checks):
        raise RuntimeError(f"Artifact reload failed: {artifact_checks}")

    del (
        outputs,
        returned_states,
        block_states,
        activations,
        cuda_batch,
        single_outputs,
        single_states,
        single_block_states,
        single_activations,
        single_cuda,
    )
    captures.clear()
    single_block_32.clear()
    throughput = benchmark_throughput(
        model,
        tokenizer,
        benchmark_rows,
        batch_size=extraction_policy["batch_size"],
    )

    report = {
        "stage": 4,
        "seed": 42,
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256_file(args.dataset),
            "rows": len(rows),
            "smoke_rows": len(smoke_rows),
            "smoke_split": "train",
            "modified": False,
        },
        "downstream_extraction_policy": extraction_policy,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "pyyaml": yaml.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_mib": to_mib(torch.cuda.get_device_properties(0).total_memory),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "torchaudio_visible": importlib.util.find_spec("torchaudio") is not None,
        },
        "model": {
            "id": args.model,
            "requested_revision": args.revision,
            "resolved_revision": hub_info.sha,
            "top_config_class": type(top_config).__name__,
            "top_config_architectures": top_config.architectures,
            "text_config_class": type(text_config).__name__,
            "loaded_model_class": type(model).__name__,
            "loaded_config_commit_hash": getattr(model.config, "_commit_hash", None),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "language_model_blocks": len(blocks),
            "hidden_size": model.config.hidden_size,
            "parameter_dtypes": parameter_dtypes,
            "parameter_devices": parameter_devices,
            "device_map": getattr(model, "hf_device_map", None),
            "loading_info": {
                "missing_keys": missing_keys,
                "mismatched_keys": mismatched_keys,
                "unexpected_key_count": len(unexpected_keys),
                "unexpected_key_prefix_counts": dict(
                    Counter(key.split(".", 1)[0] for key in unexpected_keys)
                ),
                "error_messages": error_messages,
            },
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id,
            "pad_token": tokenizer.pad_token,
            "eos_token_id": tokenizer.eos_token_id,
            "add_generation_prompt": True,
            "system_message_used": False,
            "batch_shape": list(batch["input_ids"].shape),
            "sum_mask_minus_one_is_final_index": True,
        },
        "prompts": prompt_reports,
        "hidden_states": {
            "api_returned_count": 33,
            "api_returned_shapes": returned_shapes,
            "api_returned_dtype": returned_dtype,
            "api_mapping": [
                {
                    "returned_index": 0,
                    "source": "exact output of model.model.embed_tokens",
                },
                *[
                    {
                        "returned_index": index,
                        "source": f"exact raw output of model.model.layers[{index - 1}]",
                        "transformer_block_1_based": index,
                    }
                    for index in range(1, 32)
                ],
                {
                    "returned_index": 32,
                    "source": "exact output of model.model.norm after raw block 32",
                },
            ],
            "embedding_included": True,
            "raw_block_32_included": False,
            "returned_index_0_vs_embedding": embedding_comparison,
            "returned_indices_1_to_31_exactly_match_blocks_1_to_31": all(
                match["exact"] for match in first_31_block_matches
            ),
            "returned_index_32_vs_terminal_norm": final_norm_comparison,
            "returned_index_32_vs_raw_block_32": returned_last_vs_block_32,
            "stored_mapping": [
                {
                    "stored_index": index,
                    "transformer_block_1_based": index + 1,
                    "source": f"raw hook output of model.model.layers[{index}]",
                }
                for index in range(32)
            ],
            "stored_activation_shape": list(cpu_activations.shape),
            "stored_activation_dtype": "float32",
            "all_finite": True,
        },
        "batched_vs_single": {
            "stable_id": smoke_rows[0]["stable_id"],
            "strict_elementwise_rtol": 2e-2,
            "strict_elementwise_atol": 2e-2,
            "strict_elementwise_allclose": elementwise_allclose,
            "minimum_cosine_required": minimum_cosine_required,
            "maximum_relative_l2_allowed": maximum_relative_l2_allowed,
            "agreement_criterion_passed": batch_single_agreement,
            "max_abs_difference": float(absolute_difference.max().item()),
            "mean_abs_difference": float(absolute_difference.mean().item()),
            "minimum_layer_cosine_similarity": float(cosine_by_layer.min().item()),
            "maximum_layer_cosine_similarity": float(cosine_by_layer.max().item()),
            "maximum_layer_relative_l2": float(relative_l2_by_layer.max().item()),
            "max_abs_difference_by_layer": max_abs_by_layer.tolist(),
            "mean_abs_difference_by_layer": mean_abs_by_layer.tolist(),
            "cosine_similarity_by_layer": cosine_by_layer.tolist(),
            "relative_l2_by_layer": relative_l2_by_layer.tolist(),
        },
        "artifact": {
            "path": str(args.artifact),
            "sha256": sha256_file(args.artifact),
            **artifact_checks,
        },
        "gpu_memory": {
            "after_model_load": load_memory,
            "after_smoke_forward": smoke_forward_memory,
        },
        "throughput_benchmark": throughput,
        "forward": {
            "inference_mode": True,
            "generated_answers": False,
            "use_cache": False,
            "output_hidden_states": True,
            "return_dict": True,
            "logits_to_keep": 1,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, default=json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()

