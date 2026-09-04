from collections import Counter

import numpy as np

from src.prepare_data_v2 import (
    assign_pair_splits,
    globally_match,
    restrict_common_character_support,
)


def row(identifier, characters, words):
    return {
        "stable_id": identifier,
        "character_count": characters,
        "word_count": words,
    }


def test_common_support_restricts_both_sides_inclusively():
    eval_rows = [row("e1", 5, 1), row("e2", 10, 2), row("e3", 20, 3)]
    deploy_rows = [row("d1", 8, 1), row("d2", 10, 2), row("d3", 30, 4)]
    eval_kept, deploy_kept, support = restrict_common_character_support(
        eval_rows, deploy_rows
    )
    assert support == (8, 20)
    assert [item["stable_id"] for item in eval_kept] == ["e2", "e3"]
    assert [item["stable_id"] for item in deploy_kept] == ["d1", "d2"]


def test_global_assignment_is_unique_and_uses_all_pooled_candidates():
    anchors = [row("e1", 10, 2), row("e2", 30, 6)]
    deploy = [row("d1", 11, 2), row("d2", 29, 6), row("d3", 20, 4)]
    pooled = anchors + [row("unused_eval", 100, 20)] + deploy
    matches, scaler = globally_match(anchors, deploy, pooled)
    assert len(matches) == 2
    assert len({match[1]["stable_id"] for match in matches}) == 2
    assert {match[1]["stable_id"] for match in matches} == {"d1", "d2"}
    np.testing.assert_allclose(scaler["pooled_scaler_mean"], [200 / 6, 40 / 6])


def test_pair_split_and_folds_have_exact_counts():
    pairs = [{"pair_id": str(index)} for index in range(300)]
    assign_pair_splits(pairs, seed=np.random.SeedSequence(42))
    assert Counter(pair["split"] for pair in pairs) == {"train": 210, "test": 90}
    assert Counter(
        pair["cv_fold"] for pair in pairs if pair["split"] == "train"
    ) == {0: 42, 1: 42, 2: 42, 3: 42, 4: 42}
    assert all(pair["cv_fold"] is None for pair in pairs if pair["split"] == "test")
