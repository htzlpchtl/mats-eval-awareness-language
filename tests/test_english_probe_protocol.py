import csv
from collections import Counter

import numpy as np
import pytest

from src.probe_utils import (
    assert_no_overlap,
    independent_prediction_auc,
    independent_selected_layer,
    quadrant_score_summaries,
    write_csv,
)
from src.run_english_probes import assert_fold_balance


def make_training_rows():
    rows = []
    quadrants = ("bench_eval", "casual_eval", "bench_deploy", "casual_deploy")
    for quadrant in quadrants:
        for fold in range(5):
            for index in range(42):
                rows.append(
                    {
                        "stable_id": f"{quadrant}_{fold}_{index}",
                        "quadrant": quadrant,
                        "cv_fold": fold,
                    }
                )
    return rows


def test_probe_b_fold_balance_is_exactly_42_per_quadrant():
    rows = make_training_rows()
    observed = assert_fold_balance(
        rows, ("bench_eval", "casual_eval", "bench_deploy", "casual_deploy")
    )
    assert len(observed) == 20
    assert set(observed.values()) == {42}


def test_overlap_guard_rejects_any_reused_id():
    assert_no_overlap(["train_a", "train_b"], ["test_a"])
    with pytest.raises(RuntimeError, match="overlap"):
        assert_no_overlap(["train_a", "reused"], ["reused", "test_a"])


def test_saved_csv_independently_recovers_layer_argmax_and_auc(tmp_path):
    layer_path = tmp_path / "layers.csv"
    write_csv(
        layer_path,
        [
            {"layer_index": 0, "mean_cv_auroc": 0.6},
            {"layer_index": 1, "mean_cv_auroc": 0.9},
            {"layer_index": 2, "mean_cv_auroc": 0.7},
        ],
    )
    assert independent_selected_layer(layer_path) == (1, 0.9)

    prediction_path = tmp_path / "predictions.csv"
    write_csv(
        prediction_path,
        [
            {"evaluation": "test", "context_label": 0, "decision_score": -2.0},
            {"evaluation": "test", "context_label": 0, "decision_score": -1.0},
            {"evaluation": "test", "context_label": 1, "decision_score": 1.0},
            {"evaluation": "test", "context_label": 1, "decision_score": 2.0},
        ],
    )
    assert independent_prediction_auc(prediction_path, "test") == 1.0


def test_quadrant_summary_uses_fixed_threshold_predictions():
    rows = [
        {
            "quadrant": "bench_eval",
            "decision_score": -1.0,
            "predicted_class": 0,
            "context_label": 1,
        },
        {
            "quadrant": "bench_eval",
            "decision_score": 3.0,
            "predicted_class": 1,
            "context_label": 1,
        },
    ]
    result = quadrant_score_summaries(rows)["bench_eval"]
    assert result["n"] == 2
    assert result["mean_decision_score"] == 1.0
    assert result["median_decision_score"] == 1.0
    assert result["proportion_classified_evaluation"] == 0.5
    assert result["proportion_correct"] == 0.5
