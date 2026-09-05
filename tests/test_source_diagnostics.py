import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.run_source_diagnostics import run_tfidf_cv, tfidf_pipeline


def test_tfidf_baseline_keeps_vectorizer_inside_pipeline():
    pipeline = tfidf_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    classifier = pipeline.named_steps["classifier"]
    assert classifier.l1_ratio == 0.0
    assert classifier.C == 1.0
    assert classifier.max_iter == 2000
    assert classifier.random_state == 42


def test_tfidf_cv_uses_exact_fixed_fold_partitions():
    rows = []
    for fold in range(5):
        for label in (0, 1):
            for index in range(3):
                rows.append(
                    {
                        "model_facing_text": (
                            f"shared words {'evaluation' if label else 'deployment'} "
                            f"fold{fold} item{index}"
                        ),
                        "context_label": label,
                        "cv_fold": fold,
                    }
                )
    fold_rows, summary = run_tfidf_cv(rows)
    assert len(fold_rows) == 5
    assert all(row["n_fit"] == 24 for row in fold_rows)
    assert all(row["n_validation"] == 6 for row in fold_rows)
    assert all(row["validation_positives"] == 3 for row in fold_rows)
    assert all(row["validation_negatives"] == 3 for row in fold_rows)
    assert summary["mean_cv_auroc"] == 1.0
