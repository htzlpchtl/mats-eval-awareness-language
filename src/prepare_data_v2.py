"""Create pairwise length-matched v2 data and training-only diagnostics."""

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
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.prepare_data import (
    deduplicate_lowest_row,
    stable_id,
    write_jsonl,
)
from src.utils import make_probe_pipeline


FORMAT_PAIRS = {
    "benchmark": ("bench_eval", "bench_deploy"),
    "casual": ("casual_eval", "casual_deploy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/selected/english_selected_v2.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/selected/english_selected_v2_manifest.json"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/metrics/checkpoint2b_matching_audit.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/metrics/CHECKPOINT_2B.md"),
    )
    return parser.parse_args()


def feature_vector(text: str) -> tuple[int, int]:
    return len(text), len(text.split())


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise ValueError("cannot summarize empty values")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std_population": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def restrict_common_character_support(
    eval_rows: list[dict[str, Any]], deploy_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int]]:
    lower = max(
        min(row["character_count"] for row in eval_rows),
        min(row["character_count"] for row in deploy_rows),
    )
    upper = min(
        max(row["character_count"] for row in eval_rows),
        max(row["character_count"] for row in deploy_rows),
    )
    if lower > upper:
        raise ValueError("eval and deploy pools have no overlapping character support")
    return (
        [row for row in eval_rows if lower <= row["character_count"] <= upper],
        [row for row in deploy_rows if lower <= row["character_count"] <= upper],
        (lower, upper),
    )


def globally_match(
    anchors: list[dict[str, Any]],
    deploy_candidates: list[dict[str, Any]],
    pooled_candidates: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], dict[str, Any]]:
    pooled = np.asarray(
        [
            [row["character_count"], row["word_count"]]
            for row in pooled_candidates
        ],
        dtype=float,
    )
    scaler = StandardScaler().fit(pooled)
    anchor_features = scaler.transform(
        [[row["character_count"], row["word_count"]] for row in anchors]
    )
    deploy_features = scaler.transform(
        [
            [row["character_count"], row["word_count"]]
            for row in deploy_candidates
        ]
    )
    distances = np.linalg.norm(
        anchor_features[:, None, :] - deploy_features[None, :, :], axis=2
    )
    anchor_indices, deploy_indices = linear_sum_assignment(distances)
    if len(anchor_indices) != len(anchors):
        raise AssertionError("assignment did not match every anchor")
    matches = [
        (
            anchors[int(anchor_index)],
            deploy_candidates[int(deploy_index)],
            float(distances[anchor_index, deploy_index]),
        )
        for anchor_index, deploy_index in zip(
            anchor_indices, deploy_indices, strict=True
        )
    ]
    return matches, {
        "pooled_scaler_mean": [float(value) for value in scaler.mean_],
        "pooled_scaler_scale": [float(value) for value in scaler.scale_],
    }


def assign_pair_splits(
    pairs: list[dict[str, Any]], *, seed: np.random.SeedSequence
) -> None:
    if len(pairs) != 300:
        raise ValueError("expected exactly 300 matched pairs")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    train_positions = order[:210]
    test_positions = order[210:]
    fold_order = rng.permutation(train_positions)
    fold_by_position = {
        int(position): int(rank // 42)
        for rank, position in enumerate(fold_order)
    }
    for position in train_positions:
        pairs[int(position)]["split"] = "train"
        pairs[int(position)]["cv_fold"] = fold_by_position[int(position)]
    for position in test_positions:
        pairs[int(position)]["split"] = "test"
        pairs[int(position)]["cv_fold"] = None


def source_pools(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    dataset_config = config["dataset"]
    exclusions = {
        name: set(indices)
        for name, indices in dataset_config.get("manually_excluded_rows", {}).items()
    }
    pools = {}
    for quadrant, quadrant_config in dataset_config["quadrants"].items():
        filename = f"{quadrant_config['config']}/train-00000-of-00001.parquet"
        local_path = hf_hub_download(
            repo_id=dataset_config["id"],
            repo_type="dataset",
            filename=filename,
            revision=dataset_config["revision"],
        )
        raw_rows = pq.read_table(local_path).to_pylist()
        eligible, duplicate_rows = deduplicate_lowest_row(
            raw_rows,
            quadrant_config["model_facing_field"],
            exclusions.get(quadrant, set()),
        )
        converted = []
        for original_row_index, row in eligible:
            text = row[quadrant_config["model_facing_field"]]
            chars, words = feature_vector(text)
            converted.append(
                {
                    "stable_id": stable_id(quadrant, original_row_index, text),
                    "quadrant": quadrant,
                    "context_label": int(quadrant_config["context_label"]),
                    "format_label": int(quadrant_config["format_label"]),
                    "original_row_index": original_row_index,
                    "source_id": row.get("id") or None,
                    "source": row.get("source"),
                    "routing_label": row.get("label"),
                    "model_facing_text": text,
                    "dataset_length": int(row["length"]),
                    "character_count": chars,
                    "word_count": words,
                }
            )
        pools[quadrant] = converted
        if len(converted) != len(raw_rows) - len(exclusions.get(quadrant, set())) - len(duplicate_rows):
            raise AssertionError("eligible-pool accounting failed")
    return pools


def training_length_cv(records: list[dict[str, Any]]) -> dict[str, Any]:
    training = [record for record in records if record["split"] == "train"]
    features = np.asarray(
        [
            [
                record["character_count"],
                record["word_count"],
                record["dataset_length"],
            ]
            for record in training
        ],
        dtype=float,
    )
    labels = np.asarray([record["context_label"] for record in training])
    folds = np.asarray([record["cv_fold"] for record in training])
    fold_aucs = []
    for fold in range(5):
        validation = folds == fold
        model = clone(make_probe_pipeline(seed=42))
        model.fit(features[~validation], labels[~validation])
        scores = model.decision_function(features[validation])
        fold_aucs.append(float(roc_auc_score(labels[validation], scores)))
    return {
        "features": [
            "character_count",
            "whitespace_word_count",
            "dataset_length_field",
        ],
        "note": "dataset_length_field is identical to character_count",
        "training_rows": len(training),
        "validation_rows_per_fold": [int((folds == fold).sum()) for fold in range(5)],
        "fold_auroc": fold_aucs,
        "mean_cv_auroc": float(np.mean(fold_aucs)),
        "test_partition_accessed": False,
    }


def render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Checkpoint 2B: pairwise length-matched v2 dataset",
            "",
            "The v1 pilot is retained but retired. No classifier was fit or evaluated on the v2 test partition.",
            "",
            "```json",
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["seed"])
    pools = source_pools(config)
    format_seeds = np.random.SeedSequence(seed).spawn(len(FORMAT_PAIRS) * 2)
    all_pairs = []
    format_audits = {}

    for format_index, (format_name, (eval_name, deploy_name)) in enumerate(
        FORMAT_PAIRS.items()
    ):
        eval_pool, deploy_pool, support = restrict_common_character_support(
            pools[eval_name], pools[deploy_name]
        )
        anchor_rng = np.random.default_rng(format_seeds[2 * format_index])
        anchor_positions = anchor_rng.choice(len(eval_pool), size=300, replace=False)
        anchors = [eval_pool[int(position)] for position in anchor_positions]
        matches, scaler_audit = globally_match(
            anchors, deploy_pool, eval_pool + deploy_pool
        )
        pairs = []
        for eval_row, deploy_row, distance in matches:
            pair_digest = hashlib.sha256(
                f"{eval_row['stable_id']}__{deploy_row['stable_id']}".encode("utf-8")
            ).hexdigest()[:12]
            pairs.append(
                {
                    "pair_id": f"{format_name}__{pair_digest}",
                    "format": format_name,
                    "eval": eval_row,
                    "deploy": deploy_row,
                    "matching_distance": distance,
                    "absolute_character_difference": abs(
                        eval_row["character_count"] - deploy_row["character_count"]
                    ),
                    "absolute_word_difference": abs(
                        eval_row["word_count"] - deploy_row["word_count"]
                    ),
                }
            )
        assign_pair_splits(pairs, seed=format_seeds[2 * format_index + 1])
        all_pairs.extend(pairs)
        worst = sorted(
            pairs, key=lambda pair: pair["matching_distance"], reverse=True
        )[:10]
        distance_values = np.asarray(
            [pair["matching_distance"] for pair in pairs], dtype=float
        )
        q1, q3 = np.percentile(distance_values, [25, 75])
        poor_match_threshold = float(q3 + 1.5 * (q3 - q1))
        poor_matches = sorted(
            (
                pair
                for pair in pairs
                if pair["matching_distance"] > poor_match_threshold
            ),
            key=lambda pair: pair["matching_distance"],
            reverse=True,
        )

        def match_record(pair: dict[str, Any]) -> dict[str, Any]:
            return {
                "pair_id": pair["pair_id"],
                "distance": pair["matching_distance"],
                "eval_stable_id": pair["eval"]["stable_id"],
                "deploy_stable_id": pair["deploy"]["stable_id"],
                "eval_character_count": pair["eval"]["character_count"],
                "deploy_character_count": pair["deploy"]["character_count"],
                "eval_word_count": pair["eval"]["word_count"],
                "deploy_word_count": pair["deploy"]["word_count"],
            }

        format_audits[format_name] = {
            "eval_quadrant": eval_name,
            "deploy_quadrant": deploy_name,
            "common_character_support_inclusive": list(support),
            "available_after_common_support": {
                "eval": len(eval_pool),
                "deploy": len(deploy_pool),
            },
            "selected_eval_anchors": len(anchors),
            "matched_deploy_rows": len({pair["deploy"]["stable_id"] for pair in pairs}),
            "matching_feature_scaler": scaler_audit,
            "matching_distance": summarize(
                [pair["matching_distance"] for pair in pairs]
            ),
            "absolute_within_pair_character_difference": summarize(
                [pair["absolute_character_difference"] for pair in pairs]
            ),
            "absolute_within_pair_word_difference": summarize(
                [pair["absolute_word_difference"] for pair in pairs]
            ),
            "poor_match_rule": "distance > Q3 + 1.5 * IQR",
            "poor_match_threshold": poor_match_threshold,
            "poor_match_count": len(poor_matches),
            "poor_matches": [match_record(pair) for pair in poor_matches],
            "ten_largest_distance_matches": [match_record(pair) for pair in worst],
        }

    records = []
    for pair in all_pairs:
        for side in ("eval", "deploy"):
            record = dict(pair[side])
            record.update(
                {
                    "pair_id": pair["pair_id"],
                    "pair_format": pair["format"],
                    "split": pair["split"],
                    "cv_fold": pair["cv_fold"],
                }
            )
            records.append(record)
    records.sort(key=lambda row: (row["quadrant"], row["original_row_index"]))

    assert len(records) == 1200
    assert len({record["stable_id"] for record in records}) == 1200
    assert len({record["model_facing_text"] for record in records}) == 1200
    assert Counter(record["quadrant"] for record in records) == {
        quadrant: 300 for quadrant in pools
    }
    for pair in all_pairs:
        pair_records = [record for record in records if record["pair_id"] == pair["pair_id"]]
        assert len(pair_records) == 2
        assert len({record["split"] for record in pair_records}) == 1
        assert len({record["cv_fold"] for record in pair_records}) == 1
    for format_name in FORMAT_PAIRS:
        format_pairs = [pair for pair in all_pairs if pair["format"] == format_name]
        assert Counter(pair["split"] for pair in format_pairs) == {"train": 210, "test": 90}
        assert Counter(pair["cv_fold"] for pair in format_pairs if pair["split"] == "train") == {
            fold: 42 for fold in range(5)
        }

    output_sha256 = write_jsonl(args.output, records)
    training_records = [record for record in records if record["split"] == "train"]
    training_distributions = {
        quadrant: {
            "character_count": summarize(
                [record["character_count"] for record in training_records if record["quadrant"] == quadrant]
            ),
            "word_count": summarize(
                [record["word_count"] for record in training_records if record["quadrant"] == quadrant]
            ),
            "dataset_length": summarize(
                [record["dataset_length"] for record in training_records if record["quadrant"] == quadrant]
            ),
        }
        for quadrant in pools
    }
    training_pair_differences = {
        format_name: {
            "absolute_character_difference": summarize(
                [pair["absolute_character_difference"] for pair in all_pairs if pair["format"] == format_name and pair["split"] == "train"]
            ),
            "absolute_word_difference": summarize(
                [pair["absolute_word_difference"] for pair in all_pairs if pair["format"] == format_name and pair["split"] == "train"]
            ),
        }
        for format_name in FORMAT_PAIRS
    }
    manifest = {
        "version": "v2_final_candidate",
        "dataset_id": config["dataset"]["id"],
        "dataset_revision": config["dataset"]["revision"],
        "seed": seed,
        "selected_data_path": str(args.output),
        "selected_data_sha256": output_sha256,
        "stable_id_format": "{quadrant}__{original_row_index}__{sha256(text)[:12]}",
        "pair_id_format": (
            "{format}__{sha256(eval_stable_id + '__' + deploy_stable_id)[:12]}"
        ),
        "counts": {
            "rows": len(records),
            "pairs": len(all_pairs),
            "rows_by_quadrant": dict(Counter(record["quadrant"] for record in records)),
            "pairs_by_split_and_format": {
                f"{format_name}:{split}": sum(
                    pair["format"] == format_name and pair["split"] == split
                    for pair in all_pairs
                )
                for format_name in FORMAT_PAIRS
                for split in ("train", "test")
            },
        },
    }
    audit = {
        "checkpoint": "2B",
        "manifest": manifest,
        "format_matching": format_audits,
        "training_only_length_distributions": training_distributions,
        "training_only_pair_differences": training_pair_differences,
        "training_only_length_classifier_cv": training_length_cv(records),
        "test_partition_policy": (
            "Assigned and integrity-checked only; no classifier or distribution "
            "metric was computed using v2 test rows."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(audit))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
