from collections import Counter

from src.run_tfidf_selection_sensitivity import (
    SAMPLE_PER_QUADRANT,
    assign_balanced_folds,
    sample_unmatched_training,
    selection_digest,
)


def make_rows(quadrant: str, count: int = 230):
    label = int(quadrant.endswith("eval"))
    return [
        {
            "stable_id": f"{quadrant}_{index}",
            "quadrant": quadrant,
            "context_label": label,
            "model_facing_text": f"{quadrant} example {index}",
        }
        for index in range(count)
    ]


def test_assign_balanced_folds_has_42_rows_per_fold():
    import numpy as np

    rows = assign_balanced_folds(
        make_rows("bench_eval", SAMPLE_PER_QUADRANT), np.random.default_rng(42)
    )
    assert Counter(row["cv_fold"] for row in rows) == Counter(
        {fold: 42 for fold in range(5)}
    )
    assert all(row["split"] == "train" for row in rows)


def test_unmatched_sample_is_balanced_unique_and_deterministic():
    quadrants = ("bench_eval", "bench_deploy", "casual_eval", "casual_deploy")
    pools = {quadrant: make_rows(quadrant) for quadrant in quadrants}
    first = sample_unmatched_training(pools, 42)
    second = sample_unmatched_training(pools, 42)
    assert len(first) == 840
    assert len({row["stable_id"] for row in first}) == 840
    assert Counter(row["quadrant"] for row in first) == Counter(
        {quadrant: 210 for quadrant in quadrants}
    )
    assert selection_digest(first) == selection_digest(second)
    for fold in range(5):
        assert Counter(
            row["quadrant"] for row in first if row["cv_fold"] == fold
        ) == Counter({quadrant: 42 for quadrant in quadrants})


def test_different_seed_changes_unmatched_selection():
    quadrants = ("bench_eval", "bench_deploy", "casual_eval", "casual_deploy")
    pools = {quadrant: make_rows(quadrant) for quadrant in quadrants}
    assert selection_digest(sample_unmatched_training(pools, 42)) != selection_digest(
        sample_unmatched_training(pools, 43)
    )
