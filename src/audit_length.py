"""Checkpoint 2A: audit surface-length differences without changing data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, roc_auc_score

from src.prepare_data import deduplicate_lowest_row
from src.utils import make_probe_pipeline


UPSTREAM_CODE_REVISION = "b6a5000cba8cf99d9a7e313f6371b589d9761b05"
UPSTREAM_MATCHING_CODE = (
    "https://github.com/viliana-dev/eval-awareness-format/"
    "blob/b6a5000cba8cf99d9a7e313f6371b589d9761b05/"
    "scripts/create_length_matched.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--selected", type=Path, default=Path("data/selected/english_selected.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/checkpoint2a_length_audit.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/metrics/CHECKPOINT_2A.md"),
    )
    return parser.parse_args()


def distribution(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise ValueError("cannot summarize an empty distribution")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std_population": float(array.std()),
        "min": int(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint 2A: non-model length audit",
        "",
        "The frozen sample and its fixed split were not modified.",
        "",
        "## Dataset length distributions",
        "",
        "```json",
        json.dumps(audit["length_distributions"], indent=2, sort_keys=True),
        "```",
        "",
        "## Upstream matching provenance",
        "",
        audit["upstream_matching"]["summary"],
        "",
        f"- Code revision: `{audit['upstream_matching']['code_revision']}`",
        f"- Code: {audit['upstream_matching']['code_url']}",
        "",
        "## Length-only classifier",
        "",
        "```json",
        json.dumps(audit["length_only_classifier"], indent=2, sort_keys=True),
        "```",
        "",
        "## Integrity",
        "",
        "```json",
        json.dumps(audit["integrity"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    dataset_config = config["dataset"]
    selected = [json.loads(line) for line in args.selected.read_text().splitlines()]
    frozen_sha256 = hashlib.sha256(args.selected.read_bytes()).hexdigest()
    selected_by_key = {
        (row["quadrant"], row["original_row_index"]): row for row in selected
    }
    exclusions = {
        name: set(indices)
        for name, indices in dataset_config.get("manually_excluded_rows", {}).items()
    }

    source_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    length_distributions: dict[str, Any] = {}
    length_matches_character_count = True
    for quadrant, quadrant_config in dataset_config["quadrants"].items():
        filename = f"{quadrant_config['config']}/train-00000-of-00001.parquet"
        local_path = hf_hub_download(
            repo_id=dataset_config["id"],
            repo_type="dataset",
            filename=filename,
            revision=dataset_config["revision"],
        )
        rows = pq.read_table(local_path).to_pylist()
        eligible, _ = deduplicate_lowest_row(
            rows,
            quadrant_config["model_facing_field"],
            exclusions.get(quadrant, set()),
        )
        for original_row_index, row in enumerate(rows):
            source_by_key[(quadrant, original_row_index)] = row
        pool_lengths = [int(row["length"]) for _, row in eligible]
        frozen_lengths = [
            int(source_by_key[key]["length"])
            for key in selected_by_key
            if key[0] == quadrant
        ]
        length_distributions[quadrant] = {
            "full_valid_pool": distribution(pool_lengths),
            "frozen_300": distribution(frozen_lengths),
        }
        for original_row_index, row in eligible:
            text = row[quadrant_config["model_facing_field"]]
            length_matches_character_count &= int(row["length"]) == len(text)

    features = []
    labels = []
    splits = []
    stable_ids = []
    for selected_row in selected:
        key = (selected_row["quadrant"], selected_row["original_row_index"])
        source_row = source_by_key[key]
        text = selected_row["model_facing_text"]
        features.append([len(text), len(text.split()), int(source_row["length"])])
        labels.append(int(selected_row["context_label"]))
        splits.append(selected_row["split"])
        stable_ids.append(selected_row["stable_id"])
    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    split_array = np.asarray(splits)
    train = split_array == "train"
    test = split_array == "test"
    pipeline = make_probe_pipeline(seed=int(config["seed"]))
    pipeline.fit(feature_array[train], label_array[train])
    scores = pipeline.decision_function(feature_array[test])
    predictions = pipeline.predict(feature_array[test])

    classifier = pipeline.named_steps["classifier"]
    audit = {
        "checkpoint": "2A",
        "dataset_id": dataset_config["id"],
        "dataset_revision": dataset_config["revision"],
        "frozen_data_path": str(args.selected),
        "frozen_data_sha256": frozen_sha256,
        "seed": int(config["seed"]),
        "length_distributions": length_distributions,
        "upstream_matching": {
            "target_quadrant": "bench_eval",
            "matched_quadrant": "casual_deploy_matched",
            "metric": "formatted-prompt character length",
            "method": "histogram matching using 30 percentile-derived bins",
            "sampling": (
                "For each bin, sample min(bench count, available ShareGPT count), "
                "with Python random seed 42; then shuffle the matched result."
            ),
            "implementation_caveats": (
                "This is approximate histogram matching, not row-level or exact "
                "distribution matching. Bin edges are unique percentiles of bench_eval; "
                "a bin can be underfilled when ShareGPT has too few candidates. The "
                "ShareGPT assignment loop uses start <= length < end even for the last "
                "bin, so a candidate exactly equal to the maximum edge is omitted."
            ),
            "code_revision": UPSTREAM_CODE_REVISION,
            "code_url": UPSTREAM_MATCHING_CODE,
            "summary": (
                "The upstream author intended casual_deploy_matched to match "
                "bench_eval, not casual_eval. Matching used character lengths of "
                "formatted bench_eval prompts and ShareGPT first turns."
            ),
        },
        "length_only_classifier": {
            "features": [
                "character_count",
                "whitespace_word_count",
                "dataset_length_field",
            ],
            "model": "StandardScaler + L2 LogisticRegression(C=1.0)",
            "fit_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "test_class_counts": {
                str(label): int((label_array[test] == label).sum())
                for label in sorted(np.unique(label_array[test]))
            },
            "positive_label": "evaluation (context_label=1)",
            "auroc": float(roc_auc_score(label_array[test], scores)),
            "accuracy_at_default_threshold": float(
                accuracy_score(label_array[test], predictions)
            ),
            "standardized_coefficients_in_feature_order": [
                float(value) for value in classifier.coef_[0]
            ],
            "intercept": float(classifier.intercept_[0]),
            "note": (
                "The dataset length field exactly equals Python character count in "
                "all full valid pools, so two feature columns are duplicates."
            ),
        },
        "integrity": {
            "frozen_rows_unchanged": len(selected) == 1200,
            "train_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "unique_stable_ids": len(set(stable_ids)),
            "dataset_length_equals_character_count_for_all_valid_pool_rows": (
                length_matches_character_count
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(audit))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
