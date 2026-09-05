"""Recover and independently audit the completed Stage 5 English artifact."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.extract_activations import (
    EXPECTED_ROWS,
    FROZEN_ENGLISH_V2,
    FROZEN_ENGLISH_V2_SHA256,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    N_LAYERS,
    QUADRANTS,
    REQUIRED_POLICY,
    checkpoint_path,
    contiguous_quadrant_segments,
    json_default,
    load_frozen_rows,
    render_and_measure,
    sha256_file,
    token_length_summary,
    verify_activation_artifact,
)


DEFAULT_ARTIFACT = Path("artifacts/activations/stage5_english_v2.npz")
DEFAULT_MANIFEST = Path("results/metrics/stage5_english_manifest.json")
DEFAULT_REPORT = Path("results/metrics/stage5_summary.json")


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def expected_layer_mapping() -> list[dict[str, Any]]:
    return [
        {
            "stored_index": index,
            "transformer_block_1_based": index + 1,
            "module": f"model.model.layers[{index}]",
            "source": "raw forward-hook output at final rightmost non-padding token",
        }
        for index in range(N_LAYERS)
    ]


def verify_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    artifact: Path,
    artifact_sha256: str,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_rows = [
        {
            "activation_row": index,
            "stable_id": row["stable_id"],
            "quadrant": row["quadrant"],
            "split": row["split"],
            "cv_fold": row["cv_fold"],
            "pair_id": row["pair_id"],
        }
        for index, row in enumerate(rows)
    ]
    checks = {
        "artifact_path_exact": manifest["artifact"]["path"] == str(artifact),
        "artifact_sha256_exact": manifest["artifact"]["sha256"] == artifact_sha256,
        "artifact_shape_exact": manifest["artifact"]["shape"]
        == [EXPECTED_ROWS, N_LAYERS, HIDDEN_SIZE],
        "dataset_path_exact": manifest["dataset"]["path"] == str(FROZEN_ENGLISH_V2),
        "dataset_sha256_exact": manifest["dataset"]["sha256"]
        == FROZEN_ENGLISH_V2_SHA256,
        "layer_mapping_exact": manifest["layer_mapping"] == expected_layer_mapping(),
        "row_mapping_exact": manifest["rows"] == expected_rows,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 5 manifest audit failed: {checks}")
    return checks


def main() -> None:
    from transformers import AutoConfig, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=FROZEN_ENGLISH_V2)
    parser.add_argument("--dataset-sha256", default=FROZEN_ENGLISH_V2_SHA256)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/activations"),
    )
    parser.add_argument(
        "--observed-live-device-memory-mib",
        type=float,
        default=None,
        help="Optional nvidia-smi observation made while the original run was active.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if args.dataset.resolve() != FROZEN_ENGLISH_V2.resolve():
        raise RuntimeError(f"Audit is restricted to {FROZEN_ENGLISH_V2}")
    if args.dataset_sha256 != FROZEN_ENGLISH_V2_SHA256:
        raise RuntimeError("Audit requires the pinned frozen-v2 SHA-256")
    if args.report.exists():
        raise RuntimeError(f"Refusing to overwrite existing report: {args.report}")

    rows = load_frozen_rows(args.dataset, args.dataset_sha256)
    segments = contiguous_quadrant_segments(rows)
    checkpoint_paths = [
        checkpoint_path(args.checkpoint_directory, index, quadrant)
        for index, (quadrant, _, _) in enumerate(segments, start=1)
    ]
    required_paths = [args.artifact, args.manifest, *checkpoint_paths]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing completed Stage 5 files: {missing}")

    final_checks = verify_activation_artifact(
        args.artifact,
        rows,
        np.arange(EXPECTED_ROWS, dtype=np.int32),
    )
    artifact_sha256 = sha256_file(args.artifact)
    manifest_checks = verify_manifest(
        args.manifest, rows, args.artifact, artifact_sha256
    )

    quadrant_reports = []
    checkpoints_match_final = True
    with np.load(args.artifact, allow_pickle=False) as final_artifact:
        final_activations = final_artifact["activations"]
        for segment_index, ((quadrant, start, stop), path) in enumerate(
            zip(segments, checkpoint_paths, strict=True), start=1
        ):
            segment_rows = rows[start:stop]
            checkpoint_checks = verify_activation_artifact(
                path,
                segment_rows,
                np.arange(start, stop, dtype=np.int32),
            )
            with np.load(path, allow_pickle=False) as checkpoint:
                values_match = bool(
                    np.array_equal(
                        checkpoint["activations"], final_activations[start:stop]
                    )
                )
            checkpoints_match_final = checkpoints_match_final and values_match
            quadrant_reports.append(
                {
                    "segment_index": segment_index,
                    "quadrant": quadrant,
                    "rows": stop - start,
                    "source_row_start": start,
                    "source_row_stop_exclusive": stop,
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": sha256_file(path),
                    "checkpoint_size_bytes": path.stat().st_size,
                    "checkpoint_mtime_utc": iso_mtime(path),
                    "checkpoint_reload": checkpoint_checks,
                    "values_exactly_match_final_artifact": values_match,
                }
            )
    if not checkpoints_match_final:
        raise RuntimeError("One or more quadrant checkpoints differs from the final artifact")

    top_config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if not hasattr(top_config, "text_config"):
        raise RuntimeError("Pinned Qwen config has no text_config")
    text_config = top_config.text_config
    if (
        text_config.num_hidden_layers != N_LAYERS
        or text_config.hidden_size != HIDDEN_SIZE
    ):
        raise RuntimeError("Pinned Qwen architecture differs during recovery audit")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _rendered, token_lengths = render_and_measure(tokenizer, rows)
    model_limits = [int(text_config.max_position_embeddings)]
    if 0 < int(tokenizer.model_max_length) < 1_000_000_000:
        model_limits.append(int(tokenizer.model_max_length))
    length_summary = token_length_summary(token_lengths, min(model_limits))
    if length_summary["number_exceeding_checked_model_limit"]:
        raise RuntimeError("Recovered token-length audit found an over-limit prompt")

    checkpoint_mtimes = [path.stat().st_mtime for path in checkpoint_paths]
    observable_intervals = {
        "checkpoint_1_to_2_seconds": checkpoint_mtimes[1] - checkpoint_mtimes[0],
        "checkpoint_2_to_3_seconds": checkpoint_mtimes[2] - checkpoint_mtimes[1],
        "checkpoint_3_to_4_seconds": checkpoint_mtimes[3] - checkpoint_mtimes[2],
        "checkpoint_4_to_final_artifact_seconds": args.artifact.stat().st_mtime
        - checkpoint_mtimes[3],
        "final_artifact_to_manifest_seconds": args.manifest.stat().st_mtime
        - args.artifact.stat().st_mtime,
    }
    report = {
        "stage": 5,
        "checkpoint": 5,
        "language": "english",
        "report_status": "recovered_after_summary_serialization_failure",
        "dataset": {
            "path": str(args.dataset),
            "expected_sha256": args.dataset_sha256,
            "sha256_after_recovery_audit": sha256_file(args.dataset),
            "rows": len(rows),
            "modified": False,
        },
        "model": {
            "id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_revision_verified_before_original_run": MODEL_REVISION,
            "loaded_class_verified_before_original_run": "Qwen3_5ForCausalLM",
            "language_model_blocks": int(text_config.num_hidden_layers),
            "hidden_size": int(text_config.hidden_size),
            "dtype": "bfloat16",
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
        "preflight_40": {
            "rows": 40,
            "quadrant_counts": {quadrant: 10 for quadrant in sorted(QUADRANTS)},
            "selection": "10 evenly spaced manifest positions per quadrant",
            "completed_before_full_extraction": True,
            "exact_shape_and_finite_assertions_passed_in_original_process": True,
        },
        "full_extraction": {
            "rows": EXPECTED_ROWS,
            "batches": EXPECTED_ROWS // REQUIRED_POLICY["batch_size"],
            "batch_size": REQUIRED_POLICY["batch_size"],
            "quadrant_counts": dict(Counter(row["quadrant"] for row in rows)),
            "activation_shape": [EXPECTED_ROWS, N_LAYERS, HIDDEN_SIZE],
            "all_values_finite": final_checks["all_finite"],
            "full_extraction_seconds": None,
            "prompts_per_second": None,
        },
        "quadrants": quadrant_reports,
        "layer_mapping": expected_layer_mapping(),
        "artifact": {
            "path": str(args.artifact),
            "sha256": artifact_sha256,
            "size_bytes": args.artifact.stat().st_size,
            "mtime_utc": iso_mtime(args.artifact),
            "shape": [EXPECTED_ROWS, N_LAYERS, HIDDEN_SIZE],
            "dtype": "float32",
            "reload_checks": final_checks,
            "quadrant_checkpoints_exactly_match": checkpoints_match_final,
            "manifest_path": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "manifest_checks": manifest_checks,
        },
        "runtime": {
            "exact_extraction_runtime_available": False,
            "exact_prompts_per_second_available": False,
            "reason": (
                "The original process completed extraction and artifact verification, "
                "then failed serializing set-valued model loading metadata before "
                "writing its summary. The full extraction was intentionally not rerun."
            ),
            "recovered_second_resolution_file_intervals": observable_intervals,
            "recovery_audit_seconds": time.perf_counter() - started,
        },
        "gpu_memory": {
            "exact_torch_peak_available": False,
            "reason": "Peak telemetry was lost with the original in-memory summary.",
            "observed_live_nvidia_smi_process_memory_mib": (
                args.observed_live_device_memory_mib
            ),
            "stage4_same_policy_peak_allocated_mib": 17427.6318359375,
        },
        "checks": {
            "exactly_1200_rows": True,
            "activation_shape_exact": final_checks["shape"]
            == [EXPECTED_ROWS, N_LAYERS, HIDDEN_SIZE],
            "all_values_finite": final_checks["all_finite"],
            "stable_ids_and_order_exact": final_checks["metadata_exact"],
            "quadrants_and_splits_exact": final_checks["metadata_exact"],
            "layer_numbering_explicit_and_correct": manifest_checks[
                "layer_mapping_exact"
            ],
            "quadrant_checkpoints_exactly_match_final": checkpoints_match_final,
            "dataset_unchanged": sha256_file(args.dataset)
            == FROZEN_ENGLISH_V2_SHA256,
            "stage6_started": False,
        },
        "implementation_issues": [
            {
                "issue": (
                    "The original final JSON serialization omitted the Stage 4 "
                    "set-aware JSON fallback."
                ),
                "impact": (
                    "Activation artifacts and manifest were already fully written and "
                    "verified; exact runtime, prompts/sec, and torch peak telemetry "
                    "were not persisted."
                ),
                "fix": (
                    "The extractor now uses a tested set- and NumPy-aware JSON "
                    "serializer. No activation extraction was rerun."
                ),
            }
        ],
        "warnings": [
            "Hugging Face Hub access was unauthenticated.",
            (
                "Exact Stage 5 runtime, prompts/sec, and torch peak VRAM are unavailable "
                "because of the reporting-only failure."
            ),
        ],
        "stage6_started": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
