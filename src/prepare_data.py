"""Freeze selected prompts and deterministic split assignments (Stage 2)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download

from src.utils import make_quadrant_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/selected/english_selected.jsonl")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/selected/english_selected_manifest.json"),
    )
    parser.add_argument(
        "--checkpoint-report",
        type=Path,
        default=Path("data/selected/CHECKPOINT_2.md"),
    )
    return parser.parse_args()


def short_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def stable_id(quadrant: str, original_row_index: int, text: str) -> str:
    return f"{quadrant}__{original_row_index}__{short_text_hash(text)}"


def deduplicate_lowest_row(
    rows: list[dict[str, Any]], text_field: str, excluded_rows: set[int]
) -> tuple[list[tuple[int, dict[str, Any]]], list[int]]:
    """Apply explicit exclusions and retain the first exact prompt occurrence."""

    seen: set[str] = set()
    eligible: list[tuple[int, dict[str, Any]]] = []
    duplicate_rows: list[int] = []
    for row_index, row in enumerate(rows):
        if row_index in excluded_rows:
            continue
        text = row[text_field]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"invalid {text_field} at original row {row_index}")
        if text in seen:
            duplicate_rows.append(row_index)
            continue
        seen.add(text)
        eligible.append((row_index, row))
    return eligible, duplicate_rows


def select_rows(
    rows: list[tuple[int, dict[str, Any]]], sample_size: int, rng: np.random.Generator
) -> list[tuple[int, dict[str, Any]]]:
    if len(rows) < sample_size:
        raise ValueError(f"only {len(rows)} eligible rows for sample of {sample_size}")
    positions = rng.choice(len(rows), size=sample_size, replace=False)
    return sorted((rows[int(position)] for position in positions), key=lambda item: item[0])


def numeric_summary(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    content = "".join(
        json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def render_checkpoint(manifest: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint 2: frozen English dataset and split",
        "",
        f"- Dataset: `{manifest['dataset_id']}`",
        f"- Revision: `{manifest['dataset_revision']}`",
        f"- Seed: {manifest['seed']}",
        f"- Frozen data SHA-256: `{manifest['selected_data_sha256']}`",
        f"- Stable ID: `{manifest['stable_id_format']}`",
        "- No model activations were extracted.",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(manifest["counts"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Selection audit",
        "",
        "```json",
        json.dumps(
            manifest["selection_audit"], indent=2, ensure_ascii=False, sort_keys=True
        ),
        "```",
        "",
        "## Selected-sample statistics",
        "",
        "Standard deviation uses the population definition (`ddof=0`).",
        "",
        "```json",
        json.dumps(manifest["statistics"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
    ]
    for quadrant, examples in manifest[
        "first_ten_selected_by_original_row_order"
    ].items():
        lines.extend(["", f"## First ten selected: {quadrant}", ""])
        for example in examples:
            lines.extend(
                [
                    "```json",
                    json.dumps(example, indent=2, ensure_ascii=False, sort_keys=True),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    dataset_config = config["dataset"]
    seed = int(config["seed"])
    quadrant_names = sorted(dataset_config["quadrants"])
    quadrant_seeds = np.random.SeedSequence(seed).spawn(len(quadrant_names))
    exclusions = {
        name: set(indices)
        for name, indices in dataset_config.get("manually_excluded_rows", {}).items()
    }

    selected_records: list[dict[str, Any]] = []
    selection_audit: dict[str, Any] = {}
    for quadrant, quadrant_seed in zip(quadrant_names, quadrant_seeds, strict=True):
        quadrant_config = dataset_config["quadrants"][quadrant]
        source_path = (
            f"{quadrant_config['config']}/train-00000-of-00001.parquet"
        )
        local_path = hf_hub_download(
            repo_id=dataset_config["id"],
            repo_type="dataset",
            filename=source_path,
            revision=dataset_config["revision"],
        )
        rows = pq.read_table(local_path).to_pylist()
        text_field = quadrant_config["model_facing_field"]
        excluded = exclusions.get(quadrant, set())
        if any(index < 0 or index >= len(rows) for index in excluded):
            raise ValueError(f"out-of-range manual exclusion in {quadrant}")
        eligible, duplicate_rows = deduplicate_lowest_row(rows, text_field, excluded)
        chosen = select_rows(
            eligible,
            int(dataset_config["sample_per_quadrant"]),
            np.random.default_rng(quadrant_seed),
        )
        selection_audit[quadrant] = {
            "source_path": source_path,
            "source_row_count": len(rows),
            "model_facing_field": text_field,
            "manually_excluded_original_rows": sorted(excluded),
            "deduplicated_original_rows": duplicate_rows,
            "eligible_row_count": len(eligible),
            "selected_row_count": len(chosen),
        }
        for row_index, row in chosen:
            text = row[text_field]
            record = {
                "stable_id": stable_id(quadrant, row_index, text),
                "quadrant": quadrant,
                "context_label": int(quadrant_config["context_label"]),
                "format_label": int(quadrant_config["format_label"]),
                "original_row_index": row_index,
                "source_id": row.get("id") or None,
                "source": row.get("source"),
                "routing_label": row.get("label"),
                "model_facing_text": text,
            }
            selected_records.append(record)

    ids_by_quadrant = {
        quadrant: [
            record["stable_id"]
            for record in selected_records
            if record["quadrant"] == quadrant
        ]
        for quadrant in quadrant_names
    }
    split_records = make_quadrant_splits(
        ids_by_quadrant,
        train_per_quadrant=int(dataset_config["train_per_quadrant"]),
        test_per_quadrant=int(dataset_config["test_per_quadrant"]),
        n_folds=int(dataset_config["cv_folds"]),
        seed=seed,
    )
    split_by_id = {record.stable_id: record for record in split_records}
    for record in selected_records:
        assignment = split_by_id[record["stable_id"]]
        record["split"] = assignment.split
        record["cv_fold"] = assignment.cv_fold

    selected_records.sort(
        key=lambda record: (record["quadrant"], record["original_row_index"])
    )
    ids = [record["stable_id"] for record in selected_records]
    texts = [record["model_facing_text"] for record in selected_records]
    assert len(selected_records) == 1200
    assert len(set(ids)) == 1200
    assert len(set(texts)) == 1200
    assert Counter(record["quadrant"] for record in selected_records) == Counter(
        {quadrant: 300 for quadrant in quadrant_names}
    )
    assert Counter(record["split"] for record in selected_records) == Counter(
        {"train": 840, "test": 360}
    )
    for quadrant in quadrant_names:
        assert sum(
            record["quadrant"] == quadrant and record["split"] == "train"
            for record in selected_records
        ) == 210
        assert sum(
            record["quadrant"] == quadrant and record["split"] == "test"
            for record in selected_records
        ) == 90
        for fold in range(5):
            assert sum(
                record["quadrant"] == quadrant and record["cv_fold"] == fold
                for record in selected_records
            ) == 42

    file_sha256 = write_jsonl(args.output, selected_records)
    statistics = {}
    examples = {}
    for quadrant in quadrant_names:
        quadrant_records = [
            record for record in selected_records if record["quadrant"] == quadrant
        ]
        char_lengths = [len(record["model_facing_text"]) for record in quadrant_records]
        word_counts = [
            len(record["model_facing_text"].split()) for record in quadrant_records
        ]
        source_values = [
            record["source"] or record["routing_label"]
            for record in quadrant_records
            if record["source"] or record["routing_label"]
        ]
        statistics[quadrant] = {
            "character_length": numeric_summary(char_lengths),
            "word_count": {
                key: value
                for key, value in numeric_summary(word_counts).items()
                if key in {"mean", "median"}
            },
            "source_distribution": dict(sorted(Counter(source_values).items())),
        }
        examples[quadrant] = quadrant_records[:10]

    manifest = {
        "stage": 2,
        "dataset_id": dataset_config["id"],
        "dataset_revision": dataset_config["revision"],
        "seed": seed,
        "stable_id_format": (
            "{quadrant}__{original_row_index}__"
            "{sha256(model_facing_text)[:12]}"
        ),
        "selected_data_path": str(args.output),
        "selected_data_sha256": file_sha256,
        "selection_audit": selection_audit,
        "counts": {
            "total": len(selected_records),
            "by_quadrant": dict(Counter(r["quadrant"] for r in selected_records)),
            "by_split": dict(Counter(r["split"] for r in selected_records)),
            "training_by_fold_and_quadrant": {
                f"{fold}:{quadrant}": sum(
                    record["quadrant"] == quadrant and record["cv_fold"] == fold
                    for record in selected_records
                )
                for fold in range(5)
                for quadrant in quadrant_names
            },
        },
        "statistics": statistics,
        "first_ten_selected_by_original_row_order": examples,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    )
    args.checkpoint_report.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_report.write_text(render_checkpoint(manifest))
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
