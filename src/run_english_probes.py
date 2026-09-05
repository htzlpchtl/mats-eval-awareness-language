"""Run frozen-protocol English Probe A and Probe B, then stop at Checkpoint 7."""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.extract_activations import (
    FROZEN_ENGLISH_V2,
    FROZEN_ENGLISH_V2_SHA256,
    render_and_measure,
)
from src.probe_utils import (
    MODEL_ID,
    MODEL_REVISION,
    QUADRANT_ORDER,
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
    read_csv,
    save_pipeline,
    sha256_file,
    write_csv,
    write_json,
)
from src.utils import make_probe_pipeline


def rows_at(rows: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [rows[index] for index in indices]


def labels_for(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["context_label"] for row in rows], dtype=np.int8)


def ids_for(rows: list[dict[str, Any]]) -> list[str]:
    return [row["stable_id"] for row in rows]


def assert_fold_balance(
    rows: list[dict[str, Any]], quadrants: tuple[str, ...]
) -> dict[str, int]:
    observed = Counter((row["cv_fold"], row["quadrant"]) for row in rows)
    expected = Counter(
        {(fold, quadrant): 42 for fold in range(5) for quadrant in quadrants}
    )
    if observed != expected:
        raise RuntimeError(f"Training fold/quadrant balance failed: {observed}")
    return {
        f"fold_{fold}:{quadrant}": count
        for (fold, quadrant), count in sorted(observed.items())
    }


def plot_layer_curve(
    layer_rows: list[dict[str, Any]], selected_index: int, title: str, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = np.asarray([row["layer_number"] for row in layer_rows])
    means = np.asarray([row["mean_cv_auroc"] for row in layer_rows])
    stds = np.asarray([row["std_cv_auroc"] for row in layer_rows])
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(layers, means, marker="o", markersize=3, linewidth=1.4)
    axis.fill_between(layers, means - stds, means + stds, alpha=0.2)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.axvline(selected_index + 1, color="tab:red", linestyle=":", linewidth=1)
    axis.set(
        xlabel="Transformer block (1-based)",
        ylabel="Mean five-fold CV AUROC",
        title=title,
        ylim=(0.0, 1.0),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_quadrant_scores(predictions: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [
        [row["decision_score"] for row in predictions if row["quadrant"] == quadrant]
        for quadrant in QUADRANT_ORDER
    ]
    if any(len(item) != 90 for item in values):
        raise RuntimeError("Probe B score figure requires 90 test rows per quadrant")
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    parts = axis.violinplot(values, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_alpha(0.45)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(1, 5), QUADRANT_ORDER, rotation=15)
    axis.set(
        ylabel="Probe B decision score",
        title="Probe B held-out English scores (n=90 per quadrant)",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_and_verify_probe_outputs(
    *,
    fold_path: Path,
    layer_path: Path,
    prediction_path: Path,
    fold_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    selected_index: int,
    metrics_by_evaluation: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    write_csv(fold_path, fold_rows)
    write_csv(layer_path, layer_rows)
    write_csv(prediction_path, prediction_rows)
    csv_index, csv_mean = independent_selected_layer(layer_path)
    if csv_index != selected_index:
        raise RuntimeError("Independent saved-layer argmax differs")
    aucs = {}
    for name, metrics in metrics_by_evaluation.items():
        auc = independent_prediction_auc(prediction_path, name)
        if not np.isclose(auc, metrics["auroc"], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"Independent saved-prediction AUROC differs for {name}")
        aucs[name] = auc
    return {
        "selected_layer_index": csv_index,
        "selected_mean_cv_auroc": csv_mean,
        "prediction_aurocs": aucs,
        "all_match": True,
    }


def run_probe_a(
    rows: list[dict[str, Any]], activations: np.ndarray, output_root: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    train_indices = [
        index
        for index, row in enumerate(rows)
        if row["split"] == "train"
        and row["quadrant"] in {"bench_eval", "casual_deploy"}
    ]
    ordinary_indices = [
        index
        for index, row in enumerate(rows)
        if row["split"] == "test"
        and row["quadrant"] in {"bench_eval", "casual_deploy"}
    ]
    crossed_indices = [
        index
        for index, row in enumerate(rows)
        if row["split"] == "test"
        and row["quadrant"] in {"casual_eval", "bench_deploy"}
    ]
    train_rows = rows_at(rows, train_indices)
    ordinary_rows = rows_at(rows, ordinary_indices)
    crossed_rows = rows_at(rows, crossed_indices)
    if (len(train_rows), len(ordinary_rows), len(crossed_rows)) != (420, 180, 180):
        raise RuntimeError("Probe A partition counts failed")
    assert_no_overlap(ids_for(train_rows), ids_for(ordinary_rows), ids_for(crossed_rows))
    fold_balance = assert_fold_balance(
        train_rows, ("bench_eval", "casual_deploy")
    )
    train_labels = labels_for(train_rows)
    folds = np.asarray([row["cv_fold"] for row in train_rows], dtype=np.int8)
    fold_rows, layer_rows, selected_index, cv_warnings = layerwise_cv(
        activations[train_indices],
        train_labels,
        folds,
        probe_name="probe_a",
        seed=42,
    )
    pipeline, fit_seconds, fit_warnings = fit_final_pipeline(
        activations[train_indices], train_labels, selected_index, seed=42
    )
    ordinary_metrics, ordinary_predictions = evaluate(
        pipeline,
        activations[ordinary_indices],
        labels_for(ordinary_rows),
        ordinary_rows,
        selected_index,
        evaluation_name="in_distribution",
    )
    crossed_metrics, crossed_predictions = evaluate(
        pipeline,
        activations[crossed_indices],
        labels_for(crossed_rows),
        crossed_rows,
        selected_index,
        evaluation_name="crossed_format",
    )
    fold_path = output_root / "metrics/probe_a_cv_folds.csv"
    layer_path = output_root / "metrics/probe_a_layer_cv.csv"
    prediction_path = output_root / "predictions/probe_a_english.csv"
    verification = save_and_verify_probe_outputs(
        fold_path=fold_path,
        layer_path=layer_path,
        prediction_path=prediction_path,
        fold_rows=fold_rows,
        layer_rows=layer_rows,
        prediction_rows=ordinary_predictions + crossed_predictions,
        selected_index=selected_index,
        metrics_by_evaluation={
            "in_distribution": ordinary_metrics,
            "crossed_format": crossed_metrics,
        },
    )
    pipeline_path = Path("artifacts/probes/probe_a_english.joblib")
    pipeline_metadata = {
        "probe": "A",
        "language": "english",
        "layer_index": selected_index,
        "layer_number": selected_index + 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_sha256": FROZEN_ENGLISH_V2_SHA256,
        "activation_sha256": STAGE5_ARTIFACT_SHA256,
        "training_ids": ids_for(train_rows),
    }
    pipeline_hash = save_pipeline(pipeline_path, pipeline, pipeline_metadata)
    figure_path = output_root / "figures/probe_a_cv_auc.png"
    plot_layer_curve(
        layer_rows, selected_index, "Probe A training-only layer selection", figure_path
    )
    selected_folds = [
        row["validation_auroc"]
        for row in fold_rows
        if row["layer_index"] == selected_index
    ]
    report = {
        "stage": 6,
        "checkpoint": 6,
        "probe": "A_naive_confounded",
        "seed": 42,
        "held_out_test_discipline": {
            "layer_selection_used_training_rows_only": True,
            "scaler_fit_only_within_each_cv_training_fold": True,
            "test_rows_used_for_selection": False,
            "final_pipeline_frozen_between_both_tests": True,
            "train_test_id_overlap": 0,
        },
        "training": {
            "rows": len(train_rows),
            "quadrants": dict(Counter(row["quadrant"] for row in train_rows)),
            "classes": dict(Counter(map(int, train_labels))),
            "fold_quadrant_counts": fold_balance,
            "pipeline": pipeline_spec(pipeline),
            "cv_warnings": cv_warnings,
        },
        "layer_selection": {
            "L_A_EN_layer_index": selected_index,
            "L_A_EN_transformer_block_1_based": selected_index + 1,
            "mean_cv_auroc": layer_rows[selected_index]["mean_cv_auroc"],
            "std_cv_auroc_population": layer_rows[selected_index]["std_cv_auroc"],
            "fold_aurocs": selected_folds,
            "saved_csv_argmax_verified": verification,
        },
        "final_fit": {
            "seconds": fit_seconds,
            "iterations": int(pipeline.named_steps["classifier"].n_iter_[0]),
            "warnings": fit_warnings,
            "pipeline_path": str(pipeline_path),
            "pipeline_sha256": pipeline_hash,
        },
        "in_distribution_test": {
            **ordinary_metrics,
            "quadrants": quadrant_score_summaries(ordinary_predictions),
        },
        "crossed_format_test": {
            **crossed_metrics,
            "quadrants": quadrant_score_summaries(crossed_predictions),
        },
        "outputs": {
            "cv_folds": str(fold_path),
            "cv_layers": str(layer_path),
            "predictions": str(prediction_path),
            "figure": str(figure_path),
        },
        "total_seconds": time.perf_counter() - started,
    }
    report_path = output_root / "metrics/probe_a_summary.json"
    write_json(report_path, report)
    print(
        f"Probe A complete: block={selected_index + 1}, "
        f"ID AUROC={ordinary_metrics['auroc']:.6f}, "
        f"crossed AUROC={crossed_metrics['auroc']:.6f}",
        flush=True,
    )
    return report


def metric_from_predictions(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "n": len(labels),
        "auroc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.tolist(),
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
    }


def base_prediction(
    row: dict[str, Any], layer_index: int, score: float, probability: float, prediction: int
) -> dict[str, Any]:
    return {
        "stable_id": row["stable_id"],
        "language": "english",
        "quadrant": row["quadrant"],
        "context_label": int(row["context_label"]),
        "format_label": int(row["format_label"]),
        "split": row["split"],
        "model_revision": MODEL_REVISION,
        "layer_index": layer_index,
        "layer_number": layer_index + 1,
        "decision_score": float(score),
        "predicted_probability": float(probability),
        "predicted_class": int(prediction),
    }


def run_probe_b(
    rows: list[dict[str, Any]], activations: np.ndarray, output_root: Path
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    started = time.perf_counter()
    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    test_indices = [index for index, row in enumerate(rows) if row["split"] == "test"]
    train_rows = rows_at(rows, train_indices)
    test_rows = rows_at(rows, test_indices)
    if (len(train_rows), len(test_rows)) != (840, 360):
        raise RuntimeError("Probe B partition counts failed")
    assert_no_overlap(ids_for(train_rows), ids_for(test_rows))
    fold_balance = assert_fold_balance(train_rows, QUADRANT_ORDER)
    train_labels = labels_for(train_rows)
    test_labels = labels_for(test_rows)
    folds = np.asarray([row["cv_fold"] for row in train_rows], dtype=np.int8)
    fold_rows, layer_rows, selected_index, cv_warnings = layerwise_cv(
        activations[train_indices],
        train_labels,
        folds,
        probe_name="probe_b",
        seed=42,
    )
    pipeline, fit_seconds, fit_warnings = fit_final_pipeline(
        activations[train_indices], train_labels, selected_index, seed=42
    )
    test_metrics, test_predictions = evaluate(
        pipeline,
        activations[test_indices],
        test_labels,
        test_rows,
        selected_index,
        evaluation_name="held_out_english",
    )
    fold_path = output_root / "metrics/probe_b_cv_folds.csv"
    layer_path = output_root / "metrics/probe_b_layer_cv.csv"
    prediction_path = output_root / "predictions/probe_b_english.csv"
    verification = save_and_verify_probe_outputs(
        fold_path=fold_path,
        layer_path=layer_path,
        prediction_path=prediction_path,
        fold_rows=fold_rows,
        layer_rows=layer_rows,
        prediction_rows=test_predictions,
        selected_index=selected_index,
        metrics_by_evaluation={"held_out_english": test_metrics},
    )
    pipeline_path = Path("artifacts/probes/probe_b_english.joblib")
    pipeline_metadata = {
        "probe": "B",
        "language": "english",
        "layer_index": selected_index,
        "layer_number": selected_index + 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_sha256": FROZEN_ENGLISH_V2_SHA256,
        "activation_sha256": STAGE5_ARTIFACT_SHA256,
        "training_ids": ids_for(train_rows),
    }
    pipeline_hash = save_pipeline(pipeline_path, pipeline, pipeline_metadata)

    # Prespecified cheap shuffled-label baseline: twenty deterministic shuffles.
    x_train = activations[train_indices, selected_index]
    x_test = activations[test_indices, selected_index]
    shuffled_metrics = []
    shuffled_predictions = []
    shuffled_warnings: list[str] = []
    for shuffle_seed in range(42, 62):
        shuffled_labels = train_labels.copy()
        np.random.default_rng(shuffle_seed).shuffle(shuffled_labels)
        shuffled_pipeline = make_probe_pipeline(seed=shuffle_seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shuffled_pipeline.fit(x_train, shuffled_labels)
        warning_messages = [
            f"{warning.category.__name__}: {warning.message}" for warning in caught
        ]
        shuffled_warnings.extend(warning_messages)
        if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
            raise RuntimeError(f"Shuffled baseline did not converge: {warning_messages}")
        scores = shuffled_pipeline.decision_function(x_test)
        probabilities = shuffled_pipeline.predict_proba(x_test)[:, 1]
        predictions = shuffled_pipeline.predict(x_test)
        metrics = metric_from_predictions(test_labels, scores, predictions)
        metrics.update(
            {
                "shuffle_seed": shuffle_seed,
                "iterations": int(shuffled_pipeline.named_steps["classifier"].n_iter_[0]),
            }
        )
        shuffled_metrics.append(metrics)
        for index, row in enumerate(test_rows):
            shuffled_predictions.append(
                {
                    **base_prediction(
                        row,
                        selected_index,
                        scores[index],
                        probabilities[index],
                        predictions[index],
                    ),
                    "evaluation": "shuffled_label_baseline",
                    "shuffle_seed": shuffle_seed,
                }
            )
        print(
            f"Probe B shuffle {shuffle_seed}: test AUROC={metrics['auroc']:.6f}",
            flush=True,
        )
    shuffled_metrics_path = output_root / "metrics/probe_b_shuffled_baseline.csv"
    shuffled_predictions_path = output_root / "predictions/probe_b_shuffled_english.csv"
    write_csv(shuffled_metrics_path, shuffled_metrics)
    write_csv(shuffled_predictions_path, shuffled_predictions)
    saved_shuffled = read_csv(shuffled_predictions_path)
    for metrics in shuffled_metrics:
        seed_rows = [row for row in saved_shuffled if int(row["shuffle_seed"]) == metrics["shuffle_seed"]]
        saved_auc = roc_auc_score(
            [int(row["context_label"]) for row in seed_rows],
            [float(row["decision_score"]) for row in seed_rows],
        )
        if not np.isclose(saved_auc, metrics["auroc"], rtol=0.0, atol=1e-15):
            raise RuntimeError("Saved shuffled-label AUROC differs")
    shuffled_aurocs = np.asarray([item["auroc"] for item in shuffled_metrics])

    # Prespecified cheap length baseline, trained only on the 840 training rows.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _rendered, qwen_token_counts = render_and_measure(tokenizer, rows)
    length_features = np.asarray(
        [[row["character_count"], qwen_token_counts[index]] for index, row in enumerate(rows)],
        dtype=float,
    )
    length_pipeline = make_probe_pipeline(seed=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        length_pipeline.fit(length_features[train_indices], train_labels)
    length_warnings = [f"{warning.category.__name__}: {warning.message}" for warning in caught]
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        raise RuntimeError(f"Length baseline did not converge: {length_warnings}")
    length_scores = length_pipeline.decision_function(length_features[test_indices])
    length_probabilities = length_pipeline.predict_proba(length_features[test_indices])[:, 1]
    length_predictions = length_pipeline.predict(length_features[test_indices])
    length_metrics = metric_from_predictions(test_labels, length_scores, length_predictions)
    length_prediction_rows = []
    for index, row in enumerate(test_rows):
        length_prediction_rows.append(
            {
                **base_prediction(
                    row,
                    selected_index,
                    length_scores[index],
                    length_probabilities[index],
                    length_predictions[index],
                ),
                "evaluation": "length_only_baseline",
                "character_count": int(row["character_count"]),
                "qwen_token_count": int(qwen_token_counts[test_indices[index]]),
            }
        )
    length_predictions_path = output_root / "predictions/probe_b_length_english.csv"
    write_csv(length_predictions_path, length_prediction_rows)
    independent_length_auc = independent_prediction_auc(
        length_predictions_path, "length_only_baseline"
    )
    if not np.isclose(independent_length_auc, length_metrics["auroc"], rtol=0.0, atol=1e-15):
        raise RuntimeError("Saved length-baseline AUROC differs")

    cv_figure = output_root / "figures/probe_b_cv_auc.png"
    score_figure = output_root / "figures/probe_b_english_scores.png"
    plot_layer_curve(
        layer_rows, selected_index, "Probe B training-only layer selection", cv_figure
    )
    plot_quadrant_scores(test_predictions, score_figure)
    selected_folds = [
        row["validation_auroc"]
        for row in fold_rows
        if row["layer_index"] == selected_index
    ]
    report = {
        "stage": 7,
        "checkpoint": 7,
        "probe": "B_format_controlled",
        "seed": 42,
        "held_out_test_discipline": {
            "layer_selection_used_training_rows_only": True,
            "scaler_fit_only_within_each_cv_training_fold": True,
            "test_rows_used_for_selection": False,
            "final_pipeline_fit_on_840_training_rows_only": True,
            "train_test_id_overlap": 0,
            "test_used_only_after_layer_and_pipeline_design_frozen": True,
        },
        "training": {
            "rows": len(train_rows),
            "quadrants": dict(Counter(row["quadrant"] for row in train_rows)),
            "classes": dict(Counter(map(int, train_labels))),
            "fold_quadrant_counts": fold_balance,
            "pipeline": pipeline_spec(pipeline),
            "cv_warnings": cv_warnings,
        },
        "layer_selection": {
            "L_B_EN_layer_index": selected_index,
            "L_B_EN_transformer_block_1_based": selected_index + 1,
            "mean_cv_auroc": layer_rows[selected_index]["mean_cv_auroc"],
            "std_cv_auroc_population": layer_rows[selected_index]["std_cv_auroc"],
            "fold_aurocs": selected_folds,
            "saved_csv_argmax_verified": verification,
        },
        "final_fit": {
            "seconds": fit_seconds,
            "iterations": int(pipeline.named_steps["classifier"].n_iter_[0]),
            "warnings": fit_warnings,
            "pipeline_path": str(pipeline_path),
            "pipeline_sha256": pipeline_hash,
        },
        "held_out_english_test": {
            **test_metrics,
            "quadrants": quadrant_score_summaries(test_predictions),
        },
        "shuffled_label_baseline": {
            "seeds": list(range(42, 62)),
            "repetitions": 20,
            "test_aurocs": shuffled_aurocs.tolist(),
            "mean_test_auroc": float(shuffled_aurocs.mean()),
            "std_test_auroc_population": float(shuffled_aurocs.std(ddof=0)),
            "min_test_auroc": float(shuffled_aurocs.min()),
            "median_test_auroc": float(np.median(shuffled_aurocs)),
            "max_test_auroc": float(shuffled_aurocs.max()),
            "warnings": sorted(set(shuffled_warnings)),
            "saved_prediction_aurocs_verified": True,
        },
        "length_only_baseline": {
            **length_metrics,
            "features": ["character_count", "qwen_chat_template_token_count"],
            "trained_on_training_rows_only": True,
            "qwen_token_count_summary": {
                "min": int(np.min(qwen_token_counts)),
                "median": float(np.median(qwen_token_counts)),
                "p95": float(np.percentile(qwen_token_counts, 95)),
                "max": int(np.max(qwen_token_counts)),
            },
            "warnings": length_warnings,
            "independent_csv_auroc": independent_length_auc,
        },
        "outputs": {
            "cv_folds": str(fold_path),
            "cv_layers": str(layer_path),
            "predictions": str(prediction_path),
            "shuffled_metrics": str(shuffled_metrics_path),
            "shuffled_predictions": str(shuffled_predictions_path),
            "length_predictions": str(length_predictions_path),
            "cv_figure": str(cv_figure),
            "score_figure": str(score_figure),
        },
        "total_seconds": time.perf_counter() - started,
        "translation_started": False,
    }
    report_path = output_root / "metrics/probe_b_summary.json"
    write_json(report_path, report)
    print(
        f"Probe B complete: block={selected_index + 1}, "
        f"test AUROC={test_metrics['auroc']:.6f}, "
        f"accuracy={test_metrics['accuracy']:.6f}",
        flush=True,
    )
    return report


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
        "--checkpoint-report",
        type=Path,
        default=Path("results/metrics/checkpoint7_english_summary.json"),
    )
    args = parser.parse_args()
    expected_outputs = [
        args.output_root / "metrics/probe_a_cv_folds.csv",
        args.output_root / "metrics/probe_a_layer_cv.csv",
        args.output_root / "metrics/probe_a_summary.json",
        args.output_root / "predictions/probe_a_english.csv",
        args.output_root / "figures/probe_a_cv_auc.png",
        args.output_root / "metrics/probe_b_cv_folds.csv",
        args.output_root / "metrics/probe_b_layer_cv.csv",
        args.output_root / "metrics/probe_b_summary.json",
        args.output_root / "metrics/probe_b_shuffled_baseline.csv",
        args.output_root / "predictions/probe_b_english.csv",
        args.output_root / "predictions/probe_b_shuffled_english.csv",
        args.output_root / "predictions/probe_b_length_english.csv",
        args.output_root / "figures/probe_b_cv_auc.png",
        args.output_root / "figures/probe_b_english_scores.png",
        Path("artifacts/probes/probe_a_english.joblib"),
        Path("artifacts/probes/probe_b_english.joblib"),
        args.checkpoint_report,
    ]
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite existing probe outputs: {existing}")
    started = time.perf_counter()
    rows, activations = load_frozen_activations(
        args.dataset,
        args.activations,
        dataset_sha256=args.dataset_sha256,
        activation_sha256=args.activation_sha256,
    )
    probe_a = run_probe_a(rows, activations, args.output_root)
    probe_b = run_probe_b(rows, activations, args.output_root)
    checkpoint = {
        "checkpoint": 7,
        "scope": "English Probe A and Probe B only",
        "inputs": {
            "dataset": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "activations": str(args.activations),
            "activation_sha256": sha256_file(args.activations),
            "activation_shape": list(activations.shape),
        },
        "probe_a": {
            "selected_block_1_based": probe_a["layer_selection"]["L_A_EN_transformer_block_1_based"],
            "mean_cv_auroc": probe_a["layer_selection"]["mean_cv_auroc"],
            "in_distribution_auroc": probe_a["in_distribution_test"]["auroc"],
            "in_distribution_accuracy": probe_a["in_distribution_test"]["accuracy"],
            "crossed_format_auroc": probe_a["crossed_format_test"]["auroc"],
            "crossed_format_accuracy": probe_a["crossed_format_test"]["accuracy"],
        },
        "probe_b": {
            "selected_block_1_based": probe_b["layer_selection"]["L_B_EN_transformer_block_1_based"],
            "mean_cv_auroc": probe_b["layer_selection"]["mean_cv_auroc"],
            "held_out_auroc": probe_b["held_out_english_test"]["auroc"],
            "held_out_accuracy": probe_b["held_out_english_test"]["accuracy"],
            "quadrants": probe_b["held_out_english_test"]["quadrants"],
            "shuffled_mean_auroc": probe_b["shuffled_label_baseline"]["mean_test_auroc"],
            "shuffled_range": [
                probe_b["shuffled_label_baseline"]["min_test_auroc"],
                probe_b["shuffled_label_baseline"]["max_test_auroc"],
            ],
            "length_only_auroc": probe_b["length_only_baseline"]["auroc"],
        },
        "held_out_test_discipline_passed": True,
        "optional_geometry_run": False,
        "translation_started": False,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "total_seconds": time.perf_counter() - started,
    }
    write_json(args.checkpoint_report, checkpoint)
    print(json.dumps(checkpoint, indent=2), flush=True)


if __name__ == "__main__":
    main()
