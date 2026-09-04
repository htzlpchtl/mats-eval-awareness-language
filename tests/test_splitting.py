from collections import Counter

import pytest

from src.utils import make_quadrant_splits


QUADRANTS = ("bench_eval", "casual_eval", "bench_deploy", "casual_deploy")


def synthetic_ids():
    return {
        quadrant: [f"{quadrant}:{index}" for index in range(300)]
        for quadrant in QUADRANTS
    }


def test_exact_deterministic_quadrant_split_and_balanced_folds():
    first = make_quadrant_splits(
        synthetic_ids(), train_per_quadrant=210, test_per_quadrant=90, seed=42
    )
    second = make_quadrant_splits(
        synthetic_ids(), train_per_quadrant=210, test_per_quadrant=90, seed=42
    )
    assert first == second
    assert len(first) == 1200
    assert len({record.stable_id for record in first}) == 1200

    split_counts = Counter((record.quadrant, record.split) for record in first)
    for quadrant in QUADRANTS:
        assert split_counts[(quadrant, "train")] == 210
        assert split_counts[(quadrant, "test")] == 90

    train_ids = {record.stable_id for record in first if record.split == "train"}
    test_ids = {record.stable_id for record in first if record.split == "test"}
    assert train_ids.isdisjoint(test_ids)

    fold_counts = Counter(
        (record.quadrant, record.cv_fold)
        for record in first
        if record.split == "train"
    )
    for quadrant in QUADRANTS:
        for fold in range(5):
            assert fold_counts[(quadrant, fold)] == 42
    assert all(record.cv_fold is None for record in first if record.split == "test")


def test_rejects_duplicate_stable_ids():
    ids = synthetic_ids()
    ids["casual_eval"][0] = ids["bench_eval"][0]
    with pytest.raises(ValueError, match="globally unique"):
        make_quadrant_splits(
            ids, train_per_quadrant=210, test_per_quadrant=90, seed=42
        )

