import json
from collections import Counter

import numpy as np
import pytest

from src.extract_activations import (
    N_LAYERS,
    QUADRANTS,
    choose_preflight_indices,
    contiguous_quadrant_segments,
    extraction_metadata,
    json_default,
    save_activation_artifact,
    verify_activation_artifact,
)


def make_rows():
    rows = []
    for quadrant_index, quadrant in enumerate(QUADRANTS):
        for index in range(300):
            rows.append(
                {
                    "stable_id": f"{quadrant}__{index}",
                    "quadrant": quadrant,
                    "split": "train" if index < 210 else "test",
                    "cv_fold": index % 5 if index < 210 else None,
                    "context_label": int(quadrant in {"bench_eval", "casual_eval"}),
                    "format_label": int(quadrant in {"bench_eval", "bench_deploy"}),
                    "pair_id": f"pair__{quadrant_index}__{index}",
                    "model_facing_text": f"prompt {quadrant} {index}",
                }
            )
    return rows


def test_quadrant_segments_are_manifest_order_contiguous():
    rows = make_rows()
    assert contiguous_quadrant_segments(rows) == [
        (quadrant, index * 300, (index + 1) * 300)
        for index, quadrant in enumerate(QUADRANTS)
    ]

    rows[300]["quadrant"] = QUADRANTS[0]
    with pytest.raises(RuntimeError, match="quadrant segment"):
        contiguous_quadrant_segments(rows)


def test_preflight_has_ten_unique_rows_per_quadrant_in_manifest_order():
    rows = make_rows()
    indices = choose_preflight_indices(rows)
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices)) == 40
    assert Counter(rows[index]["quadrant"] for index in indices) == Counter(
        {quadrant: 10 for quadrant in QUADRANTS}
    )
    for quadrant in QUADRANTS:
        quadrant_indices = [
            index for index in indices if rows[index]["quadrant"] == quadrant
        ]
        assert len(quadrant_indices) == 10
        assert quadrant_indices[-1] - quadrant_indices[0] == 299


def test_metadata_makes_row_and_layer_alignment_explicit():
    rows = make_rows()[:3]
    metadata = extraction_metadata(rows, np.asarray([17, 18, 19]))
    np.testing.assert_array_equal(metadata["row_indices"], [17, 18, 19])
    np.testing.assert_array_equal(
        metadata["stable_ids"], [row["stable_id"] for row in rows]
    )
    np.testing.assert_array_equal(metadata["layer_numbers"], np.arange(1, 33))
    np.testing.assert_array_equal(metadata["layer_module_indices"], np.arange(32))
    assert metadata["layer_sources"][0] == (
        "raw hook output of model.model.layers[0]"
    )
    assert metadata["layer_sources"][-1] == (
        "raw hook output of model.model.layers[31]"
    )


def test_activation_artifact_round_trip_and_alignment(tmp_path):
    rows = make_rows()[:3]
    row_indices = np.asarray([0, 1, 2])
    activations = np.arange(3 * N_LAYERS * 4, dtype=np.float32).reshape(
        3, N_LAYERS, 4
    )
    path = tmp_path / "activations.npz"
    save_activation_artifact(
        path, rows, row_indices, activations, hidden_size=4
    )
    checks = verify_activation_artifact(
        path,
        rows,
        row_indices,
        activations,
        hidden_size=4,
    )
    assert checks == {
        "shape": [3, 32, 4],
        "dtype": "float32",
        "metadata_exact": True,
        "values_exact": True,
        "all_finite": True,
    }

    with pytest.raises(RuntimeError, match="stable_ids"):
        verify_activation_artifact(
            path,
            list(reversed(rows)),
            row_indices,
            activations,
            hidden_size=4,
        )


def test_json_default_serializes_loading_metadata_sets_and_numpy_scalars():
    encoded = json.dumps(
        {"set_value": {"z", "a"}, "numpy_value": np.int64(7)},
        default=json_default,
    )
    assert json.loads(encoded) == {
        "set_value": ["a", "z"],
        "numpy_value": 7,
    }
