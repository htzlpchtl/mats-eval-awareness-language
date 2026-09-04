"""Deterministic helpers shared by data, activation, and probe stages."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SplitRecord:
    """Bookkeeping assigned to one stable example ID."""

    stable_id: Hashable
    quadrant: str
    split: str
    cv_fold: int | None


def make_quadrant_splits(
    ids_by_quadrant: dict[str, Sequence[Hashable]],
    *,
    train_per_quadrant: int,
    test_per_quadrant: int,
    n_folds: int = 5,
    seed: int = 42,
) -> list[SplitRecord]:
    """Assign seeded train/test splits and balanced CV folds per quadrant.

    Each quadrant is independently shuffled. Training examples are shuffled a
    second time before folds are assigned in equal-sized blocks. Input IDs must
    already be stable; this function never derives identity from row order.
    """

    if train_per_quadrant % n_folds != 0:
        raise ValueError("train_per_quadrant must be divisible by n_folds")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    all_ids = [item for ids in ids_by_quadrant.values() for item in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("stable IDs must be globally unique")

    records: list[SplitRecord] = []
    seed_sequence = np.random.SeedSequence(seed)
    quadrant_seeds = seed_sequence.spawn(len(ids_by_quadrant))

    for (quadrant, source_ids), quadrant_seed in zip(
        sorted(ids_by_quadrant.items()), quadrant_seeds, strict=True
    ):
        ids = list(source_ids)
        required = train_per_quadrant + test_per_quadrant
        if len(ids) != required:
            raise ValueError(
                f"{quadrant} has {len(ids)} IDs; expected exactly {required}"
            )

        rng = np.random.default_rng(quadrant_seed)
        shuffled = np.asarray(ids, dtype=object)[rng.permutation(len(ids))].tolist()
        train_ids = shuffled[:train_per_quadrant]
        test_ids = shuffled[train_per_quadrant:]

        fold_order = np.asarray(train_ids, dtype=object)[
            rng.permutation(train_per_quadrant)
        ].tolist()
        fold_size = train_per_quadrant // n_folds
        fold_by_id = {
            stable_id: position // fold_size
            for position, stable_id in enumerate(fold_order)
        }

        records.extend(
            SplitRecord(stable_id, quadrant, "train", fold_by_id[stable_id])
            for stable_id in train_ids
        )
        records.extend(
            SplitRecord(stable_id, quadrant, "test", None)
            for stable_id in test_ids
        )

    return records


def final_nonpadding_indices(attention_mask: Any) -> Any:
    """Return final non-padding positions for left- or right-padded masks.

    Supports NumPy arrays and torch tensors without importing torch. Every row
    must contain at least one attended token. Internal mask gaps are permitted;
    the returned position is the rightmost element equal to one.
    """

    if getattr(attention_mask, "ndim", None) != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if attention_mask.shape[1] == 0:
        raise ValueError("attention_mask sequence dimension cannot be empty")

    if isinstance(attention_mask, np.ndarray):
        present = attention_mask != 0
        if not np.all(present.any(axis=1)):
            raise ValueError("every example must contain an attended token")
        reversed_positions = np.argmax(present[:, ::-1], axis=1)
        return attention_mask.shape[1] - 1 - reversed_positions

    # Torch-like path: argmax is applied after converting bool to integer for
    # compatibility across torch versions.
    present = attention_mask.ne(0)
    if not bool(present.any(dim=1).all().item()):
        raise ValueError("every example must contain an attended token")
    reversed_positions = present.flip(dims=(1,)).to(dtype=attention_mask.dtype).argmax(dim=1)
    return attention_mask.shape[1] - 1 - reversed_positions


def select_final_nonpadding(hidden_states: Any, attention_mask: Any) -> Any:
    """Select one [hidden] vector per row from [batch, sequence, hidden]."""

    if getattr(hidden_states, "ndim", None) != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("hidden_states and attention_mask batch/sequence shapes differ")

    positions = final_nonpadding_indices(attention_mask)
    if isinstance(hidden_states, np.ndarray):
        return hidden_states[np.arange(hidden_states.shape[0]), positions]

    import torch

    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, positions.to(device=hidden_states.device)]


def make_probe_pipeline(*, seed: int = 42) -> Pipeline:
    """Construct the fixed, leakage-safe probe pipeline."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    l1_ratio=0.0, C=1.0, max_iter=2000, random_state=seed
                ),
            ),
        ]
    )


def layerwise_cv_auc(
    activations: np.ndarray,
    labels: np.ndarray,
    cv_folds: np.ndarray,
    *,
    pipeline: Pipeline | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Evaluate independent pipelines at each layer using fixed fold IDs.

    `activations` must have shape [examples, layers, hidden]. A fresh clone is
    fitted on each fold, ensuring the scaler sees fold-training data only.
    Returns fold AUCs, mean AUCs, and the zero-based layer argmax.
    """

    activations = np.asarray(activations)
    labels = np.asarray(labels)
    cv_folds = np.asarray(cv_folds)
    if activations.ndim != 3:
        raise ValueError("activations must have shape [examples, layers, hidden]")
    if not (len(activations) == len(labels) == len(cv_folds)):
        raise ValueError("activation, label, and fold lengths must match")

    unique_folds = np.unique(cv_folds)
    if len(unique_folds) < 2:
        raise ValueError("at least two CV folds are required")
    base_pipeline = make_probe_pipeline(seed=seed) if pipeline is None else pipeline
    fold_aucs = np.empty((activations.shape[1], len(unique_folds)), dtype=float)

    for layer in range(activations.shape[1]):
        for column, fold in enumerate(unique_folds):
            is_validation = cv_folds == fold
            model = clone(base_pipeline)
            model.fit(activations[~is_validation, layer], labels[~is_validation])
            scores = model.decision_function(activations[is_validation, layer])
            fold_aucs[layer, column] = roc_auc_score(labels[is_validation], scores)

    mean_aucs = fold_aucs.mean(axis=1)
    return fold_aucs, mean_aucs, int(np.argmax(mean_aucs))
