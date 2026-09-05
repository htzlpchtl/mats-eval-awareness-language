"""Test whether v2 pairwise matching unusually increases TF-IDF identifiability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import yaml

from src.extract_activations import FROZEN_ENGLISH_V2, FROZEN_ENGLISH_V2_SHA256, load_frozen_rows, sha256_file
from src.prepare_data_v2 import FORMAT_PAIRS, restrict_common_character_support, source_pools
from src.run_source_diagnostics import run_tfidf_cv


SAMPLE_PER_QUADRANT = 210
SEEDS = tuple(range(42, 62))
EXPECTED_SUPPORT = {"benchmark": (103, 672), "casual": (52, 711)}


def assign_balanced_folds(rows: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    if len(rows) != SAMPLE_PER_QUADRANT:
        raise ValueError("expected 210 rows per quadrant")
    order = rng.permutation(len(rows))
    fold_by_position = {int(position): int(rank // 42) for rank, position in enumerate(order)}
    output = []
    for position, row in enumerate(rows):
        converted = dict(row)
        converted["split"] = "train"
        converted["cv_fold"] = fold_by_position[position]
        output.append(converted)
    if Counter(row["cv_fold"] for row in output) != Counter({fold: 42 for fold in range(5)}):
        raise RuntimeError("fold assignment is not balanced")
    return output


def sample_unmatched_training(
    pools: dict[str, list[dict[str, Any]]], seed: int
) -> list[dict[str, Any]]:
    quadrants = sorted(pools)
    child_seeds = np.random.SeedSequence(seed).spawn(len(quadrants))
    sampled: list[dict[str, Any]] = []
    for quadrant, child_seed in zip(quadrants, child_seeds, strict=True):
        candidates = pools[quadrant]
        if len(candidates) < SAMPLE_PER_QUADRANT:
            raise RuntimeError(f"{quadrant} has too few sampling candidates")
        rng = np.random.default_rng(child_seed)
        positions = rng.choice(len(candidates), size=SAMPLE_PER_QUADRANT, replace=False)
        chosen = [candidates[int(position)] for position in positions]
        sampled.extend(assign_balanced_folds(chosen, rng))
    if len(sampled) != 840 or len({row["stable_id"] for row in sampled}) != 840:
        raise RuntimeError("unmatched sample must have 840 unique rows")
    for fold in range(5):
        counts = Counter(row["quadrant"] for row in sampled if row["cv_fold"] == fold)
        if counts != Counter({quadrant: 42 for quadrant in quadrants}):
            raise RuntimeError(f"seed {seed} fold {fold} is not quadrant-balanced")
    return sampled


def selection_digest(rows: list[dict[str, Any]]) -> str:
    assignments = sorted((row["stable_id"], int(row["cv_fold"])) for row in rows)
    encoded = json.dumps(assignments, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    v2 = report["frozen_v2_training_tfidf"]
    random_summary = report["unmatched_repeated_sampling"]["summary"]
    formats = report["frozen_v2_format_specific_tfidf"]
    lines = [
        "# TF-IDF selection-sensitivity diagnostic",
        "",
        "All results use training rows only. Each TF-IDF vectorizer and logistic classifier was fit inside its CV training fold.",
        "",
        "| Dataset | Mean CV AUROC | Range |",
        "|---|---:|---:|",
        f"| Frozen v2 pooled | {v2['mean_cv_auroc']:.6f} | {min(v2['fold_aurocs']):.6f}-{max(v2['fold_aurocs']):.6f} across folds |",
        f"| Unmatched source-pool samples ({random_summary['seeds']} seeds) | {random_summary['mean_seed_cv_auroc']:.6f} | {random_summary['min_seed_cv_auroc']:.6f}-{random_summary['max_seed_cv_auroc']:.6f} across seed means |",
        f"| Frozen v2 benchmark only | {formats['benchmark']['mean_cv_auroc']:.6f} | {min(formats['benchmark']['fold_aurocs']):.6f}-{max(formats['benchmark']['fold_aurocs']):.6f} across folds |",
        f"| Frozen v2 casual only | {formats['casual']['mean_cv_auroc']:.6f} | {min(formats['casual']['fold_aurocs']):.6f}-{max(formats['casual']['fold_aurocs']):.6f} across folds |",
        "",
        f"Frozen-v2 minus unmatched mean: {report['comparison']['frozen_v2_minus_unmatched_mean']:.6f} AUROC.",
        f"Fraction of unmatched seed means at least as high as frozen v2: {report['comparison']['fraction_unmatched_at_least_frozen_v2']:.3f}.",
        "",
        "The unmatched samples were independently drawn per quadrant after exact deduplication/manual exclusions and restriction to the same benchmark and casual common character-length supports used by v2. Frozen held-out IDs were excluded before sampling. No pairwise matching was used.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--dataset", type=Path, default=FROZEN_ENGLISH_V2)
    parser.add_argument("--dataset-sha256", default=FROZEN_ENGLISH_V2_SHA256)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--first-seed", type=int, default=SEEDS[0])
    parser.add_argument("--report", type=Path, default=Path("results/metrics/tfidf_selection_sensitivity.json"))
    parser.add_argument("--seed-csv", type=Path, default=Path("results/metrics/tfidf_selection_sensitivity_seeds.csv"))
    parser.add_argument("--fold-csv", type=Path, default=Path("results/metrics/tfidf_selection_sensitivity_folds.csv"))
    parser.add_argument("--format-csv", type=Path, default=Path("results/metrics/tfidf_frozen_format_cv.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("results/metrics/TFIDF_SELECTION_SENSITIVITY.md"))
    args = parser.parse_args()
    if args.dataset.resolve() != FROZEN_ENGLISH_V2.resolve() or args.dataset_sha256 != FROZEN_ENGLISH_V2_SHA256:
        raise RuntimeError("diagnostic requires the frozen v2 manifest and hash")
    if args.seeds < 2:
        raise ValueError("repeated sampling requires at least two seeds")
    outputs = [args.report, args.seed_csv, args.fold_csv, args.format_csv, args.markdown]
    if any(path.exists() for path in outputs):
        raise RuntimeError("refusing to overwrite sensitivity outputs")

    started = time.perf_counter()
    rows = load_frozen_rows(args.dataset, args.dataset_sha256)
    train_rows = [row for row in rows if row["split"] == "train"]
    heldout_ids = {row["stable_id"] for row in rows if row["split"] == "test"}
    train_ids = {row["stable_id"] for row in train_rows}
    if len(train_rows) != 840 or len(heldout_ids) != 360 or train_ids & heldout_ids:
        raise RuntimeError("frozen train/test partition is invalid")

    v2_folds, v2_summary = run_tfidf_cv(train_rows)
    existing = json.loads(Path("results/metrics/source_confounding_diagnostics.json").read_text())[
        "tfidf_probe_b_training_cv"
    ]
    if not np.isclose(v2_summary["mean_cv_auroc"], existing["mean_cv_auroc"], rtol=0, atol=1e-15):
        raise RuntimeError("frozen-v2 TF-IDF rerun differs from the existing result")

    format_rows = []
    format_summaries = {}
    for format_name in ("benchmark", "casual"):
        selected = [row for row in train_rows if row["pair_format"] == format_name]
        if len(selected) != 420:
            raise RuntimeError(f"{format_name} frozen training rows must total 420")
        fold_rows, summary = run_tfidf_cv(selected)
        for row in fold_rows:
            format_rows.append({"format": format_name, **row})
        format_summaries[format_name] = summary

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    full_pools = source_pools(config)
    sampling_pools: dict[str, list[dict[str, Any]]] = {}
    pool_audit = {}
    for format_name, (eval_name, deploy_name) in FORMAT_PAIRS.items():
        eval_supported, deploy_supported, support = restrict_common_character_support(
            full_pools[eval_name], full_pools[deploy_name]
        )
        if support != EXPECTED_SUPPORT[format_name]:
            raise RuntimeError(f"{format_name} support changed: {support}")
        for quadrant, supported in ((eval_name, eval_supported), (deploy_name, deploy_supported)):
            eligible = [row for row in supported if row["stable_id"] not in heldout_ids]
            sampling_pools[quadrant] = eligible
            pool_audit[quadrant] = {
                "deduplicated_pre_v2_rows": len(full_pools[quadrant]),
                "rows_within_common_support": len(supported),
                "frozen_heldout_ids_excluded": len(supported) - len(eligible),
                "sampling_candidates": len(eligible),
                "common_character_support_inclusive": list(support),
            }

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    seed_rows = []
    fold_rows_all = []
    for seed in seeds:
        sampled = sample_unmatched_training(sampling_pools, seed)
        if {row["stable_id"] for row in sampled} & heldout_ids:
            raise RuntimeError("unmatched sample contains a frozen held-out ID")
        folds, summary = run_tfidf_cv(sampled)
        seed_rows.append({
            "seed": seed,
            "rows": len(sampled),
            "rows_per_quadrant": SAMPLE_PER_QUADRANT,
            "mean_cv_auroc": summary["mean_cv_auroc"],
            "std_fold_auroc_population": summary["std_cv_auroc_population"],
            "min_fold_auroc": min(summary["fold_aurocs"]),
            "max_fold_auroc": max(summary["fold_aurocs"]),
            "selection_and_fold_sha256": selection_digest(sampled),
            "overlap_with_frozen_train": len({row["stable_id"] for row in sampled} & train_ids),
            "overlap_with_frozen_test": 0,
        })
        for fold in folds:
            fold_rows_all.append({"seed": seed, **fold})
    seed_aurocs = np.asarray([row["mean_cv_auroc"] for row in seed_rows])
    repeated_summary = {
        "seeds": len(seeds), "seed_values": seeds,
        "mean_seed_cv_auroc": float(seed_aurocs.mean()),
        "std_seed_cv_auroc_population": float(seed_aurocs.std(ddof=0)),
        "median_seed_cv_auroc": float(np.median(seed_aurocs)),
        "min_seed_cv_auroc": float(seed_aurocs.min()),
        "max_seed_cv_auroc": float(seed_aurocs.max()),
    }
    write_csv(args.seed_csv, seed_rows)
    write_csv(args.fold_csv, fold_rows_all)
    write_csv(args.format_csv, format_rows)
    report = {
        "diagnostic": "TF-IDF sensitivity to v2 pairwise length-matching selection",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(args.dataset), "sha256": sha256_file(args.dataset), "source_id": config["dataset"]["id"], "source_revision": config["dataset"]["revision"]},
        "protocol": {
            "training_only": True,
            "heldout_labels_consulted": False,
            "frozen_heldout_ids_excluded_from_source_pool_sampling": True,
            "sample_rows_per_quadrant": SAMPLE_PER_QUADRANT,
            "fold_rows_per_quadrant": 42,
            "folds": 5,
            "pairwise_matching": False,
            "sampling": "independent uniform sampling without replacement within each quadrant and seed",
            "tfidf_fit_scope": "fresh vectorizer and classifier inside each CV training fold",
            "tfidf_pipeline": v2_summary["pipeline"],
        },
        "pool_audit": pool_audit,
        "frozen_v2_training_tfidf": {**v2_summary, "rows": 840, "existing_result_reproduced_exactly": True},
        "frozen_v2_format_specific_tfidf": format_summaries,
        "unmatched_repeated_sampling": {"summary": repeated_summary, "per_seed": seed_rows},
        "comparison": {
            "frozen_v2_minus_unmatched_mean": float(v2_summary["mean_cv_auroc"] - seed_aurocs.mean()),
            "fraction_unmatched_at_least_frozen_v2": float(np.mean(seed_aurocs >= v2_summary["mean_cv_auroc"])),
        },
        "outputs": {"seed_csv": str(args.seed_csv), "fold_csv": str(args.fold_csv), "format_csv": str(args.format_csv), "markdown": str(args.markdown)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "runtime_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown, report)
    print(json.dumps({"v2": v2_summary["mean_cv_auroc"], "unmatched": repeated_summary, "formats": {key: value["mean_cv_auroc"] for key, value in format_summaries.items()}, "comparison": report["comparison"], "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
