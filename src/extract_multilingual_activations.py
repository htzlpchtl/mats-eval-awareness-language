"""Extract held-out ES/JA activations with the frozen Stage 5 policy."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.extract_activations import (
    BATCH_SIZE,
    FROZEN_ENGLISH_V2,
    FROZEN_ENGLISH_V2_SHA256,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    N_LAYERS,
    REQUIRED_POLICY,
    extract_indices,
    json_default,
    render_and_measure,
    save_activation_artifact,
    sha256_file,
    to_mib,
    token_length_summary,
    verify_activation_artifact,
)
from src.translate_prompts import frozen_test_rows, validate_output_rows


TRANSLATIONS = {
    "spanish": Path("data/translated/spanish_test_v2.jsonl"),
    "japanese": Path("data/translated/japanese_test_v2.jsonl"),
}
ARTIFACTS = {
    "spanish": Path("artifacts/activations/stage9_spanish_test_v2.npz"),
    "japanese": Path("artifacts/activations/stage9_japanese_test_v2.npz"),
}


def load_translations(
    path: Path,
    language: str,
    sources: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    validate_output_rows(rows, sources, language)
    return rows


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
        "--report",
        type=Path,
        default=Path("results/metrics/stage9_multilingual_extraction.json"),
    )
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    program_started = time.perf_counter()

    if args.dataset.resolve() != FROZEN_ENGLISH_V2.resolve():
        raise RuntimeError("Extraction is restricted to the frozen v2 manifest")
    if args.dataset_sha256 != FROZEN_ENGLISH_V2_SHA256:
        raise RuntimeError("The frozen v2 SHA-256 is required")
    if args.model != MODEL_ID or args.revision != MODEL_REVISION:
        raise RuntimeError("The frozen Stage 5 Qwen model and revision are required")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["activation_extraction"] != REQUIRED_POLICY:
        raise RuntimeError("config.yaml differs from the frozen extraction policy")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    existing = [str(path) for path in [*ARTIFACTS.values(), args.report] if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite Stage 9 outputs: " + ", ".join(existing))

    sources = frozen_test_rows(args.dataset, args.dataset_sha256)
    source_indices = np.asarray([index for index, _row in sources], dtype=np.int32)
    rows_by_language = {
        language: load_translations(path, language, sources)
        for language, path in TRANSLATIONS.items()
    }
    translation_hashes_initial = {
        language: sha256_file(path) for language, path in TRANSLATIONS.items()
    }

    hub_info = HfApi().model_info(args.model, revision=args.revision)
    if hub_info.sha != args.revision:
        raise RuntimeError(f"Revision resolved to {hub_info.sha}, expected {args.revision}")
    top_config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    if getattr(top_config, "_commit_hash", None) != args.revision:
        raise RuntimeError("Loaded config did not resolve to the pinned revision")
    text_config = top_config.text_config
    if text_config.num_hidden_layers != N_LAYERS or text_config.hidden_size != HIDDEN_SIZE:
        raise RuntimeError("Unexpected Qwen text architecture")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prepared = {}
    model_limits = [int(text_config.max_position_embeddings)]
    if 0 < int(tokenizer.model_max_length) < 1_000_000_000:
        model_limits.append(int(tokenizer.model_max_length))
    checked_limit = min(model_limits)
    for language, rows in rows_by_language.items():
        rendered, lengths = render_and_measure(tokenizer, rows)
        summary = token_length_summary(lengths, checked_limit)
        if summary["number_exceeding_checked_model_limit"]:
            raise RuntimeError(f"{language}: prompt exceeds input limit; no truncation allowed")
        prepared[language] = (rendered, lengths, summary)

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
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise RuntimeError("Incomplete Qwen checkpoint load")
    disallowed = [
        key
        for key in loading_info.get("unexpected_keys", [])
        if not (key.startswith("model.visual.") or key.startswith("mtp."))
    ]
    if disallowed:
        raise RuntimeError(f"Unexpected checkpoint keys: {disallowed[:10]}")
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise RuntimeError(f"Unexpected model class: {type(model).__name__}")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"Unexpected parameter dtypes: {parameter_dtypes}")

    language_reports = {}
    for language, rows in rows_by_language.items():
        # Recheck all frozen inputs immediately before each extraction.
        dataset_hash_before = sha256_file(args.dataset)
        translation_hash_before = sha256_file(TRANSLATIONS[language])
        if dataset_hash_before != args.dataset_sha256:
            raise RuntimeError("Frozen English manifest changed immediately before extraction")
        if translation_hash_before != translation_hashes_initial[language]:
            raise RuntimeError(f"{language} translation manifest changed before extraction")
        rendered, lengths, length_summary = prepared[language]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        run_started = time.perf_counter()
        activations, stats = extract_indices(
            model,
            tokenizer,
            rendered,
            lengths,
            list(range(len(rows))),
            batch_size=BATCH_SIZE,
            require_consecutive=True,
        )
        extraction_seconds = time.perf_counter() - run_started
        artifact = ARTIFACTS[language]
        save_started = time.perf_counter()
        save_activation_artifact(artifact, rows, source_indices, activations)
        checks = verify_activation_artifact(artifact, rows, source_indices, activations)
        save_reload_seconds = time.perf_counter() - save_started
        language_reports[language] = {
            "translation_path": str(TRANSLATIONS[language]),
            "translation_sha256_immediately_before": translation_hash_before,
            "artifact_path": str(artifact),
            "artifact_sha256": sha256_file(artifact),
            "artifact_size_bytes": artifact.stat().st_size,
            "dataset_sha256_immediately_before": dataset_hash_before,
            "row_count": len(rows),
            "quadrant_counts": dict(Counter(row["quadrant"] for row in rows)),
            "source_manifest_row_indices_preserved": True,
            "token_lengths": length_summary,
            "extraction_seconds_measured_outer": extraction_seconds,
            "save_and_reload_seconds": save_reload_seconds,
            "peak_allocated_mib": to_mib(torch.cuda.max_memory_allocated()),
            "peak_reserved_mib": to_mib(torch.cuda.max_memory_reserved()),
            "artifact_checks": checks,
            **stats,
        }
        del activations

    final_hashes = {language: sha256_file(path) for language, path in TRANSLATIONS.items()}
    if final_hashes != translation_hashes_initial or sha256_file(args.dataset) != args.dataset_sha256:
        raise RuntimeError("A frozen input changed during extraction")
    report = {
        "stage": "minimal held-out multilingual activation extraction",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": time.perf_counter() - program_started,
        "model_id": args.model,
        "model_revision": args.revision,
        "model_revision_resolved": hub_info.sha,
        "model_class": type(model).__name__,
        "model_load_seconds": model_load_seconds,
        "model_parameter_dtype": parameter_dtypes,
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "selection": "only 360 frozen held-out rows, in filtered frozen-manifest order",
        "policy": REQUIRED_POLICY,
        "no_generation": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "datasets": datasets.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "languages": language_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=json_default))


if __name__ == "__main__":
    main()
