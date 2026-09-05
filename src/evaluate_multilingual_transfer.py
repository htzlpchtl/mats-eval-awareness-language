"""Apply frozen English Probe B to held-out EN/ES/JA block-32 activations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score

from src.extract_activations import FROZEN_ENGLISH_V2, FROZEN_ENGLISH_V2_SHA256, MODEL_REVISION, sha256_file, verify_activation_artifact
from src.probe_utils import QUADRANT_ORDER, STAGE5_ARTIFACT_SHA256, load_frozen_activations, quadrant_score_summaries
from src.translate_prompts import frozen_test_rows, validate_output_rows

PROBE_SHA256 = "c6039d72695ac36fe2c422b5d563e8f98ed90fb5b743c1e1a29f67fbc084652d"
PROBE = Path("artifacts/probes/probe_b_english.joblib")
EN_ACTIVATIONS = Path("artifacts/activations/stage5_english_v2.npz")
TRANSLATIONS = {language: Path(f"data/translated/{language}_test_v2.jsonl") for language in ("spanish", "japanese")}
ACTIVATIONS = {language: Path(f"artifacts/activations/stage9_{language}_test_v2.npz") for language in ("spanish", "japanese")}
NAMES = {"english": "EN->EN", "spanish": "EN->ES", "japanese": "EN->JA"}
LAYER_INDEX = 31


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    assignments = np.minimum(np.digitize(probabilities, np.linspace(0, 1, 11)[1:-1]), 9)
    return float(sum(
        (assignments == index).mean()
        * abs(probabilities[assignments == index].mean() - labels[assignments == index].mean())
        for index in range(10)
        if (assignments == index).any()
    ))


def evaluate_language(pipeline: Any, activations: np.ndarray, rows: list[dict[str, Any]], language: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray([row["context_label"] for row in rows], dtype=np.int8)
    features = activations[:, LAYER_INDEX]
    scores = np.asarray(pipeline.decision_function(features), dtype=np.float64)
    probabilities = np.asarray(pipeline.predict_proba(features)[:, 1], dtype=np.float64)
    predictions = np.asarray(pipeline.predict(features), dtype=np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics = {
        "evaluation": NAMES[language], "language": language, "n": len(rows),
        "positives": int(labels.sum()), "negatives": int(len(labels) - labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": {"labels": [0, 1], "matrix": matrix.tolist(), "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]), "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])},
        "score_summary": {
            "mean": float(scores.mean()), "median": float(np.median(scores)),
            "std_population": float(scores.std(ddof=0)), "min": float(scores.min()), "max": float(scores.max()),
            "mean_context_eval": float(scores[labels == 1].mean()),
            "mean_context_deploy": float(scores[labels == 0].mean()),
        },
        "fixed_threshold_calibration": {
            "decision_threshold": 0.0, "probability_threshold": 0.5,
            "mean_predicted_probability": float(probabilities.mean()),
            "observed_positive_rate": float(labels.mean()),
            "predicted_positive_rate": float(predictions.mean()),
            "brier_score": float(brier_score_loss(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "ece_10_equal_width_bins": expected_calibration_error(labels, probabilities),
        },
    }
    output = []
    for position, row in enumerate(rows):
        output.append({
            "stable_id": row["stable_id"], "manifest_row_index": row.get("manifest_row_index"),
            "language": language, "evaluation": NAMES[language], "quadrant": row["quadrant"],
            "context_label": int(row["context_label"]), "format_label": int(row["format_label"]),
            "split": row["split"], "model_revision": MODEL_REVISION,
            "layer_index": LAYER_INDEX, "layer_number": LAYER_INDEX + 1,
            "decision_score": float(scores[position]), "predicted_probability": float(probabilities[position]),
            "predicted_class": int(predictions[position]), "correct": int(predictions[position] == labels[position]),
        })
    metrics["quadrants"] = quadrant_score_summaries(output)
    return metrics, output


def paired_shift(english: list[dict[str, Any]], target: list[dict[str, Any]]) -> dict[str, float]:
    en = {row["stable_id"]: row for row in english}
    other = {row["stable_id"]: row for row in target}
    if en.keys() != other.keys():
        raise RuntimeError("Paired language prediction IDs differ")
    ids = [row["stable_id"] for row in english]
    delta = np.asarray([other[key]["decision_score"] - en[key]["decision_score"] for key in ids])
    labels = np.asarray([en[key]["context_label"] for key in ids])
    en_scores = np.asarray([en[key]["decision_score"] for key in ids])
    target_scores = np.asarray([other[key]["decision_score"] for key in ids])
    return {
        "mean_paired_decision_score_delta_vs_english": float(delta.mean()),
        "median_paired_decision_score_delta_vs_english": float(np.median(delta)),
        "std_paired_decision_score_delta_population": float(delta.std(ddof=0)),
        "mean_delta_context_eval": float(delta[labels == 1].mean()),
        "mean_delta_context_deploy": float(delta[labels == 0].mean()),
        "paired_score_pearson_r": float(np.corrcoef(en_scores, target_scores)[0, 1]),
    }


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Minimal multilingual transfer checkpoint", "", "> Probe B is source-confounded and is not a clean measure of evaluation awareness.", "", "| Transfer | AUROC | Accuracy | Mean score | Predicted eval rate | Brier | ECE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES.values():
        item = report["evaluations"][name]; cal = item["fixed_threshold_calibration"]
        lines.append(f"| {name} | {item['auroc']:.6f} | {item['accuracy']:.6f} | {item['score_summary']['mean']:.6f} | {cal['predicted_positive_rate']:.6f} | {cal['brier_score']:.6f} | {cal['ece_10_equal_width_bins']:.6f} |")
    for name in NAMES.values():
        lines += ["", f"## {name} quadrant breakdown", "", "| Quadrant | n | Mean score | Median | SD | Eval-class rate | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|"]
        for quadrant in QUADRANT_ORDER:
            item = report["evaluations"][name]["quadrants"][quadrant]
            lines.append(f"| {quadrant} | {item['n']} | {item['mean_decision_score']:.6f} | {item['median_decision_score']:.6f} | {item['std_decision_score_population']:.6f} | {item['proportion_classified_evaluation']:.6f} | {item['proportion_correct']:.6f} |")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=FROZEN_ENGLISH_V2)
    parser.add_argument("--english-activations", type=Path, default=EN_ACTIVATIONS)
    parser.add_argument("--probe", type=Path, default=PROBE)
    parser.add_argument("--predictions", type=Path, default=Path("results/metrics/stage9_multilingual_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/metrics/stage9_transfer.json"))
    parser.add_argument("--markdown", type=Path, default=Path("results/metrics/STAGE9_TRANSFER.md"))
    args = parser.parse_args()
    if sha256_file(args.probe) != PROBE_SHA256:
        raise RuntimeError("Frozen English Probe B artifact hash mismatch")
    saved = joblib.load(args.probe); pipeline = saved["pipeline"]; metadata = saved["metadata"]
    if metadata.get("layer_index") != LAYER_INDEX or metadata.get("layer_number") != 32:
        raise RuntimeError("Frozen Probe B did not select block 32")

    all_rows, all_acts = load_frozen_activations(args.dataset, args.english_activations)
    en_indices = [i for i, row in enumerate(all_rows) if row["split"] == "test"]
    en_rows = []
    for index in en_indices:
        row = dict(all_rows[index]); row["manifest_row_index"] = index; en_rows.append(row)
    rows_by_language = {"english": en_rows}; acts_by_language = {"english": all_acts[en_indices]}
    sources = frozen_test_rows(args.dataset, FROZEN_ENGLISH_V2_SHA256)
    source_indices = np.asarray([index for index, _row in sources], dtype=np.int32)
    for language in ("spanish", "japanese"):
        rows = read_jsonl(TRANSLATIONS[language]); validate_output_rows(rows, sources, language)
        verify_activation_artifact(ACTIVATIONS[language], rows, source_indices)
        with np.load(ACTIVATIONS[language], allow_pickle=False) as artifact:
            acts_by_language[language] = artifact["activations"]
        rows_by_language[language] = rows

    evaluations = {}; predictions = {}; combined = []
    for language in NAMES:
        metrics, language_predictions = evaluate_language(pipeline, acts_by_language[language], rows_by_language[language], language)
        evaluations[NAMES[language]] = metrics; predictions[language] = language_predictions; combined += language_predictions
    for language in ("spanish", "japanese"):
        evaluations[NAMES[language]]["score_shift_vs_EN->EN"] = paired_shift(predictions["english"], predictions[language])
    write_csv(args.predictions, combined)
    with args.predictions.open(encoding="utf-8", newline="") as handle:
        reloaded = list(csv.DictReader(handle))
    independent = {name: float(roc_auc_score([int(row["context_label"]) for row in reloaded if row["evaluation"] == name], [float(row["decision_score"]) for row in reloaded if row["evaluation"] == name])) for name in NAMES.values()}
    classifier = pipeline.named_steps["classifier"]; scaler = pipeline.named_steps["scaler"]
    report = {
        "checkpoint": "minimal multilingual frozen English Probe B transfer",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Source-confounded activation signal, not clean evaluation awareness. English TF-IDF CV AUROC was 0.895692 and cross-format transfer was 0.799/0.657.",
        "probe_application": "prediction only; no fit or target-language refit",
        "probe_path": str(args.probe), "probe_sha256": sha256_file(args.probe),
        "english_activation_sha256": STAGE5_ARTIFACT_SHA256,
        "dataset_sha256": FROZEN_ENGLISH_V2_SHA256, "model_revision": MODEL_REVISION,
        "layer_index": LAYER_INDEX, "layer_number": LAYER_INDEX + 1,
        "frozen_parameter_hashes": {"scaler_mean": array_sha256(scaler.mean_), "scaler_scale": array_sha256(scaler.scale_), "classifier_coef": array_sha256(classifier.coef_), "classifier_intercept": array_sha256(classifier.intercept_)},
        "inputs": {language: {"translation_sha256": sha256_file(TRANSLATIONS[language]), "activation_sha256": sha256_file(ACTIVATIONS[language])} for language in ("spanish", "japanese")},
        "evaluations": evaluations, "independent_csv_reload_aurocs": independent,
        "predictions_path": str(args.predictions), "predictions_sha256": sha256_file(args.predictions),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown, report); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
