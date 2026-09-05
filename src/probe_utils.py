"""Shared, explicit utilities for the frozen English linear probes."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from src.extract_activations import (
    FROZEN_ENGLISH_V2_SHA256,
    HIDDEN_SIZE,
    N_LAYERS,
    load_frozen_rows,
    sha256_file,
    verify_activation_artifact,
)
from src.utils import make_probe_pipeline


STAGE5_ARTIFACT_SHA256 = (
    "dbac459e5670ee9e61fa90c7e62f4e60ce305589b6d1c55ac2b041c7c90136f8"
)
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QUADRANT_ORDER = ("bench_eval", "casual_eval", "bench_deploy", "casual_deploy")


def load_frozen_activations(
    dataset_path: Path,
    activation_path: Path,
    *,
    dataset_sha256: str = FROZEN_ENGLISH_V2_SHA256,
    activation_sha256: str = STAGE5_ARTIFACT_SHA256,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load Stage 5 only after independently checking row/layer alignment."""

    if sha256_file(dataset_path) != dataset_sha256:
        raise RuntimeError("Frozen-v2 dataset SHA-256 mismatch")
    if sha256_file(activation_path) != activation_sha256:
        raise RuntimeError("Stage 5 activation artifact SHA-256 mismatch")
    rows = load_frozen_rows(dataset_path, dataset_sha256)
    expected_indices = np.arange(len(rows), dtype=np.int32)
    checks = verify_activation_artifact(activation_path, rows, expected_indices)
    if not checks["metadata_exact"] or not checks["all_finite"]:
        raise RuntimeError(f"Stage 5 activation verification failed: {checks}")
    with np.load(activation_path, allow_pickle=False) as artifact:
        activations = artifact["activations"]
    if activations.shape != (1200, N_LAYERS, HIDDEN_SIZE):
        raise RuntimeError(f"Unexpected Stage 5 shape: {activations.shape}")
    if activations.dtype != np.float32 or not np.isfinite(activations).all():
        raise RuntimeError("Stage 5 activations are not finite float32")
    return rows, activations


def assert_no_overlap(*id_groups: list[str]) -> None:
    sets = [set(group) for group in id_groups]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            overlap = sets[left] & sets[right]
            if overlap:
                raise RuntimeError(f"ID partitions overlap: {sorted(overlap)[:5]}")


def layerwise_cv(
    activations: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    probe_name: str,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[str]]:
    """Fit a fresh scaler/logistic pipeline per layer and fixed fold."""

    activations = np.asarray(activations)
    labels = np.asarray(labels, dtype=np.int8)
    folds = np.asarray(folds, dtype=np.int8)
    if activations.ndim != 3 or activations.shape[1:] != (N_LAYERS, HIDDEN_SIZE):
        raise ValueError("CV activations must have shape [rows, 32, 4096]")
    if not (len(activations) == len(labels) == len(folds)):
        raise ValueError("CV arrays differ in length")
    if set(np.unique(folds).tolist()) != set(range(5)):
        raise ValueError("CV folds must be exactly 0-4")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("CV labels must contain both binary classes")

    base = make_probe_pipeline(seed=seed)
    fold_rows: list[dict[str, Any]] = []
    warnings_seen: list[str] = []
    for layer_index in range(N_LAYERS):
        for fold in range(5):
            validation = folds == fold
            pipeline = clone(base)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(
                    activations[~validation, layer_index], labels[~validation]
                )
            elapsed = time.perf_counter() - started
            for warning in caught:
                message = f"{warning.category.__name__}: {warning.message}"
                warnings_seen.append(message)
            scores = pipeline.decision_function(
                activations[validation, layer_index]
            )
            fold_rows.append(
                {
                    "probe": probe_name,
                    "layer_index": layer_index,
                    "layer_number": layer_index + 1,
                    "fold": fold,
                    "n_fit": int((~validation).sum()),
                    "n_validation": int(validation.sum()),
                    "validation_positives": int(labels[validation].sum()),
                    "validation_negatives": int(validation.sum() - labels[validation].sum()),
                    "validation_auroc": float(
                        roc_auc_score(labels[validation], scores)
                    ),
                    "fit_seconds": elapsed,
                    "iterations": int(
                        pipeline.named_steps["classifier"].n_iter_[0]
                    ),
                }
            )
        layer_values = [
            row["validation_auroc"]
            for row in fold_rows
            if row["layer_index"] == layer_index
        ]
        print(
            f"{probe_name} layer {layer_index + 1:02d}: "
            f"mean CV AUROC={np.mean(layer_values):.6f}",
            flush=True,
        )

    summary_rows = []
    for layer_index in range(N_LAYERS):
        values = np.asarray(
            [
                row["validation_auroc"]
                for row in fold_rows
                if row["layer_index"] == layer_index
            ],
            dtype=float,
        )
        summary_rows.append(
            {
                "probe": probe_name,
                "layer_index": layer_index,
                "layer_number": layer_index + 1,
                "mean_cv_auroc": float(values.mean()),
                "std_cv_auroc": float(values.std(ddof=0)),
                "min_cv_auroc": float(values.min()),
                "max_cv_auroc": float(values.max()),
            }
        )
    selected_index = int(
        np.argmax([row["mean_cv_auroc"] for row in summary_rows])
    )
    if any("ConvergenceWarning" in item for item in warnings_seen):
        raise RuntimeError(f"Logistic regression did not converge: {warnings_seen}")
    return fold_rows, summary_rows, selected_index, sorted(set(warnings_seen))


def fit_final_pipeline(
    activations: np.ndarray,
    labels: np.ndarray,
    layer_index: int,
    *,
    seed: int = 42,
) -> tuple[Any, float, list[str]]:
    pipeline = make_probe_pipeline(seed=seed)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(activations[:, layer_index], labels)
    elapsed = time.perf_counter() - started
    warning_messages = sorted(
        {f"{warning.category.__name__}: {warning.message}" for warning in caught}
    )
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        raise RuntimeError(f"Final logistic regression did not converge: {warning_messages}")
    return pipeline, elapsed, warning_messages


def evaluate(
    pipeline: Any,
    activations: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
    layer_index: int,
    *,
    evaluation_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=np.int8)
    x = activations[:, layer_index]
    scores = np.asarray(pipeline.decision_function(x), dtype=float)
    probabilities = np.asarray(pipeline.predict_proba(x)[:, 1], dtype=float)
    predictions = np.asarray(pipeline.predict(x), dtype=np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics = {
        "evaluation": evaluation_name,
        "n": len(rows),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
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
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "stable_id": row["stable_id"],
                "language": "english",
                "evaluation": evaluation_name,
                "quadrant": row["quadrant"],
                "context_label": int(row["context_label"]),
                "format_label": int(row["format_label"]),
                "split": row["split"],
                "model_revision": MODEL_REVISION,
                "layer_index": layer_index,
                "layer_number": layer_index + 1,
                "decision_score": float(scores[index]),
                "predicted_probability": float(probabilities[index]),
                "predicted_class": int(predictions[index]),
            }
        )
    return metrics, prediction_rows


def quadrant_score_summaries(
    prediction_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for quadrant in QUADRANT_ORDER:
        selected = [row for row in prediction_rows if row["quadrant"] == quadrant]
        if not selected:
            continue
        scores = np.asarray([row["decision_score"] for row in selected], dtype=float)
        predicted = np.asarray([row["predicted_class"] for row in selected], dtype=int)
        labels = np.asarray([row["context_label"] for row in selected], dtype=int)
        output[quadrant] = {
            "n": len(selected),
            "context_label": int(labels[0]),
            "mean_decision_score": float(scores.mean()),
            "median_decision_score": float(np.median(scores)),
            "std_decision_score_population": float(scores.std(ddof=0)),
            "min_decision_score": float(scores.min()),
            "max_decision_score": float(scores.max()),
            "proportion_classified_evaluation": float(predicted.mean()),
            "proportion_correct": float((predicted == labels).mean()),
        }
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def independent_prediction_auc(path: Path, evaluation_name: str) -> float:
    rows = [
        row for row in read_csv(path) if row["evaluation"] == evaluation_name
    ]
    return float(
        roc_auc_score(
            [int(row["context_label"]) for row in rows],
            [float(row["decision_score"]) for row in rows],
        )
    )


def independent_selected_layer(path: Path) -> tuple[int, float]:
    rows = read_csv(path)
    best = max(
        rows,
        key=lambda row: (float(row["mean_cv_auroc"]), -int(row["layer_index"])),
    )
    return int(best["layer_index"]), float(best["mean_cv_auroc"])


def save_pipeline(path: Path, pipeline: Any, metadata: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reloaded = joblib.load(path)
    if reloaded["metadata"] != metadata:
        raise RuntimeError("Saved pipeline metadata changed on reload")
    return digest


def pipeline_spec(pipeline: Any) -> dict[str, Any]:
    classifier = pipeline.named_steps["classifier"]
    return {
        "steps": [name for name, _ in pipeline.steps],
        "scaler": type(pipeline.named_steps["scaler"]).__name__,
        "classifier": type(classifier).__name__,
        "regularization": "L2 via l1_ratio=0.0 under sklearn 1.9",
        "C": classifier.C,
        "solver": classifier.solver,
        "max_iter": classifier.max_iter,
        "random_state": classifier.random_state,
        "threshold": 0.5,
        "class_weight": classifier.class_weight,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
