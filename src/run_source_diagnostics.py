"""Run training-only lexical and cross-format source-confounding diagnostics."""

from __future__ import annotations

import argparse
import platform
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.extract_activations import FROZEN_ENGLISH_V2, FROZEN_ENGLISH_V2_SHA256
from src.probe_utils import (
    MODEL_ID,
    MODEL_REVISION,
    STAGE5_ARTIFACT_SHA256,
    assert_no_overlap,
    evaluate,
    fit_final_pipeline,
    independent_prediction_auc,
    independent_selected_layer,
    layerwise_cv,
    load_frozen_activations,
    pipeline_spec,
    quadrant_score_summaries,
    save_pipeline,
    sha256_file,
    write_csv,
    write_json,
)


def tfidf_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "classifier",
                LogisticRegression(
                    l1_ratio=0.0,
                    C=1.0,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def run_tfidf_cv(train_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts = np.asarray([row["model_facing_text"] for row in train_rows], dtype=object)
    labels = np.asarray([row["context_label"] for row in train_rows], dtype=np.int8)
    folds = np.asarray([row["cv_fold"] for row in train_rows], dtype=np.int8)
    fold_rows = []
    warnings_seen = []
    for fold in range(5):
        validation = folds == fold
        model = tfidf_pipeline()
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(texts[~validation].tolist(), labels[~validation])
        warning_messages = [
            f"{warning.category.__name__}: {warning.message}" for warning in caught
        ]
        warnings_seen.extend(warning_messages)
        if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
            raise RuntimeError(f"TF-IDF baseline did not converge: {warning_messages}")
        scores = model.decision_function(texts[validation].tolist())
        fold_rows.append(
            {
                "fold": fold,
                "n_fit": int((~validation).sum()),
                "n_validation": int(validation.sum()),
                "validation_positives": int(labels[validation].sum()),
                "validation_negatives": int(validation.sum() - labels[validation].sum()),
                "validation_auroc": float(roc_auc_score(labels[validation], scores)),
                "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                "iterations": int(model.named_steps["classifier"].n_iter_[0]),
                "fit_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"TF-IDF fold {fold}: AUROC={fold_rows[-1]['validation_auroc']:.6f}",
            flush=True,
        )
    aucs = np.asarray([row["validation_auroc"] for row in fold_rows])
    return fold_rows, {
        "mean_cv_auroc": float(aucs.mean()),
        "std_cv_auroc_population": float(aucs.std(ddof=0)),
        "fold_aurocs": aucs.tolist(),
        "warnings": sorted(set(warnings_seen)),
        "pipeline": {
            "tfidf": "sklearn TfidfVectorizer defaults: lowercase word unigrams, no feature cap",
            "classifier": "L2 LogisticRegression",
            "C": 1.0,
            "max_iter": 2000,
            "random_state": 42,
        },
        "fit_scope": "each vectorizer and classifier fit only on the corresponding four training folds",
    }


def run_direction(
    name: str,
    source_format: str,
    target_format: str,
    rows: list[dict[str, Any]],
    activations: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    source_indices = [
        index
        for index, row in enumerate(rows)
        if row["split"] == "train" and row["pair_format"] == source_format
    ]
    target_indices = [
        index
        for index, row in enumerate(rows)
        if row["split"] == "train" and row["pair_format"] == target_format
    ]
    source_rows = [rows[index] for index in source_indices]
    target_rows = [rows[index] for index in target_indices]
    if len(source_rows) != 420 or len(target_rows) != 420:
        raise RuntimeError(f"{name} requires 420 source and 420 target training rows")
    assert_no_overlap(
        [row["stable_id"] for row in source_rows],
        [row["stable_id"] for row in target_rows],
    )
    source_quadrants = sorted({row["quadrant"] for row in source_rows})
    fold_counts = Counter((row["cv_fold"], row["quadrant"]) for row in source_rows)
    expected = Counter(
        {(fold, quadrant): 42 for fold in range(5) for quadrant in source_quadrants}
    )
    if fold_counts != expected:
        raise RuntimeError(f"{name} source folds are not quadrant-balanced")
    labels = np.asarray([row["context_label"] for row in source_rows], dtype=np.int8)
    folds = np.asarray([row["cv_fold"] for row in source_rows], dtype=np.int8)
    fold_rows, layer_rows, selected_index, cv_warnings = layerwise_cv(
        activations[source_indices], labels, folds, probe_name=name, seed=42
    )
    pipeline, fit_seconds, fit_warnings = fit_final_pipeline(
        activations[source_indices], labels, selected_index, seed=42
    )
    target_labels = np.asarray(
        [row["context_label"] for row in target_rows], dtype=np.int8
    )
    metrics, predictions = evaluate(
        pipeline,
        activations[target_indices],
        target_labels,
        target_rows,
        selected_index,
        evaluation_name=name,
    )
    fold_path = output_root / f"metrics/{name}_cv_folds.csv"
    layer_path = output_root / f"metrics/{name}_layer_cv.csv"
    prediction_path = output_root / f"predictions/{name}.csv"
    write_csv(fold_path, fold_rows)
    write_csv(layer_path, layer_rows)
    write_csv(prediction_path, predictions)
    csv_layer, csv_mean = independent_selected_layer(layer_path)
    csv_auc = independent_prediction_auc(prediction_path, name)
    if csv_layer != selected_index or not np.isclose(
        csv_auc, metrics["auroc"], rtol=0.0, atol=1e-15
    ):
        raise RuntimeError(f"{name} saved-file verification failed")
    pipeline_path = Path(f"artifacts/probes/{name}.joblib")
    pipeline_hash = save_pipeline(
        pipeline_path,
        pipeline,
        {
            "diagnostic": name,
            "source_format": source_format,
            "target_format": target_format,
            "selected_layer_index": selected_index,
            "selected_layer_number": selected_index + 1,
            "training_ids": [row["stable_id"] for row in source_rows],
            "evaluation_ids": [row["stable_id"] for row in target_rows],
            "evaluation_partition": "train",
            "model_revision": MODEL_REVISION,
        },
    )
    return {
        "direction": name,
        "source_format": source_format,
        "target_format": target_format,
        "source_training_rows": len(source_rows),
        "target_evaluation_rows": len(target_rows),
        "evaluation_partition": "train",
        "held_out_test_rows_used": 0,
        "source_quadrant_counts": dict(Counter(row["quadrant"] for row in source_rows)),
        "target_quadrant_counts": dict(Counter(row["quadrant"] for row in target_rows)),
        "fold_quadrant_counts": {
            f"fold_{fold}:{quadrant}": count
            for (fold, quadrant), count in sorted(fold_counts.items())
        },
        "layer_selection": {
            "selected_layer_index": selected_index,
            "selected_transformer_block_1_based": selected_index + 1,
            "mean_cv_auroc": layer_rows[selected_index]["mean_cv_auroc"],
            "std_cv_auroc_population": layer_rows[selected_index]["std_cv_auroc"],
            "fold_aurocs": [
                row["validation_auroc"]
                for row in fold_rows
                if row["layer_index"] == selected_index
            ],
            "selected_from_source_training_cv_only": True,
            "saved_csv_argmax": {"layer_index": csv_layer, "mean_cv_auroc": csv_mean},
        },
        "transfer_evaluation": {
            **metrics,
            "quadrants": quadrant_score_summaries(predictions),
            "independent_csv_auroc": csv_auc,
        },
        "pipeline": pipeline_spec(pipeline),
        "final_fit_seconds": fit_seconds,
        "final_fit_warnings": fit_warnings,
        "cv_warnings": cv_warnings,
        "pipeline_path": str(pipeline_path),
        "pipeline_sha256": pipeline_hash,
        "outputs": {
            "fold_metrics": str(fold_path),
            "layer_metrics": str(layer_path),
            "predictions": str(prediction_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=FROZEN_ENGLISH_V2)
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path("artifacts/activations/stage5_english_v2.npz"),
    )
    parser.add_argument("--dataset-sha256", default=FROZEN_ENGLISH_V2_SHA256)
    parser.add_argument("--activation-sha256", default=STAGE5_ARTIFACT_SHA256)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/metrics/source_confounding_diagnostics.json"),
    )
    args = parser.parse_args()
    expected_outputs = [
        args.report,
        args.output_root / "metrics/tfidf_probe_b_cv.csv",
        args.output_root / "metrics/benchmark_to_casual_cv_folds.csv",
        args.output_root / "metrics/benchmark_to_casual_layer_cv.csv",
        args.output_root / "predictions/benchmark_to_casual.csv",
        args.output_root / "metrics/casual_to_benchmark_cv_folds.csv",
        args.output_root / "metrics/casual_to_benchmark_layer_cv.csv",
        args.output_root / "predictions/casual_to_benchmark.csv",
        Path("artifacts/probes/benchmark_to_casual.joblib"),
        Path("artifacts/probes/casual_to_benchmark.joblib"),
    ]
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite diagnostic outputs: {existing}")
    started = time.perf_counter()
    rows, activations = load_frozen_activations(
        args.dataset,
        args.activations,
        dataset_sha256=args.dataset_sha256,
        activation_sha256=args.activation_sha256,
    )
    train_rows = [row for row in rows if row["split"] == "train"]
    if len(train_rows) != 840:
        raise RuntimeError("TF-IDF baseline requires all 840 training rows")
    tfidf_folds, tfidf_summary = run_tfidf_cv(train_rows)
    tfidf_path = args.output_root / "metrics/tfidf_probe_b_cv.csv"
    write_csv(tfidf_path, tfidf_folds)
    with tfidf_path.open(encoding="utf-8") as handle:
        saved = list(__import__("csv").DictReader(handle))
    saved_mean = float(np.mean([float(row["validation_auroc"]) for row in saved]))
    if not np.isclose(saved_mean, tfidf_summary["mean_cv_auroc"], rtol=0.0, atol=1e-15):
        raise RuntimeError("Saved TF-IDF mean CV AUROC differs")
    tfidf_summary["saved_csv_mean_cv_auroc"] = saved_mean
    tfidf_summary["saved_csv_verified"] = True
    benchmark_to_casual = run_direction(
        "benchmark_to_casual", "benchmark", "casual", rows, activations, args.output_root
    )
    casual_to_benchmark = run_direction(
        "casual_to_benchmark", "casual", "benchmark", rows, activations, args.output_root
    )
    report = {
        "diagnostic": "source_provenance_and_cross_format",
        "seed": 42,
        "inputs": {
            "dataset": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "activations": str(args.activations),
            "activation_sha256": sha256_file(args.activations),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
        "implementation_leakage_audit": {
            "frozen_cv_folds_used": True,
            "cv_folds": [0, 1, 2, 3, 4],
            "examples_per_quadrant_per_validation_fold": 42,
            "scaler_fit_only_on_each_fold_training_partition": True,
            "probe_feature_input": "activation tensor slice only; no labels, IDs, quadrant, text, or metadata concatenated",
            "train_test_ids_disjoint": True,
            "probe_b_shuffled_label_mean_held_out_auroc": 0.49794598765432097,
            "held_out_rows_used_by_new_diagnostics": 0,
        },
        "tfidf_probe_b_training_cv": {
            **tfidf_summary,
            "rows": 840,
            "folds": 5,
            "held_out_test_rows_used": 0,
            "output": str(tfidf_path),
        },
        "cross_format_transfer": {
            "benchmark_to_casual": benchmark_to_casual,
            "casual_to_benchmark": casual_to_benchmark,
            "interpretation_scope": "Each direction selects and fits on one format's training rows and evaluates once on the other format's training rows; no held-out test rows are used.",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "total_seconds": time.perf_counter() - started,
        "translation_started": False,
        "optional_geometry_run": False,
    }
    write_json(args.report, report)
    print(
        "Diagnostics complete: "
        f"TF-IDF CV={tfidf_summary['mean_cv_auroc']:.6f}, "
        f"benchmark->casual={benchmark_to_casual['transfer_evaluation']['auroc']:.6f}, "
        f"casual->benchmark={casual_to_benchmark['transfer_evaluation']['auroc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
