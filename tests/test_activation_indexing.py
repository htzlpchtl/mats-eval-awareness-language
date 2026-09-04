import numpy as np
import pytest

from src.utils import final_nonpadding_indices, select_final_nonpadding


def test_selects_rightmost_nonpadding_vector_for_mixed_padding():
    hidden = np.arange(3 * 6 * 2).reshape(3, 6, 2)
    mask = np.array(
        [
            [1, 1, 1, 0, 0, 0],  # right padding; final token index 2
            [0, 0, 1, 1, 1, 1],  # left padding; final token index 5
            [1, 1, 1, 1, 1, 1],  # no padding; final token index 5
        ]
    )

    np.testing.assert_array_equal(final_nonpadding_indices(mask), [2, 5, 5])
    selected = select_final_nonpadding(hidden, mask)
    np.testing.assert_array_equal(selected, hidden[[0, 1, 2], [2, 5, 5]])
    assert not np.array_equal(selected[0], hidden[0, -1])


def test_rejects_empty_attention_row():
    with pytest.raises(ValueError, match="attended token"):
        final_nonpadding_indices(np.array([[1, 0], [0, 0]]))


def test_torch_tensor_path_when_torch_is_available():
    torch = pytest.importorskip("torch")
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    selected = select_final_nonpadding(hidden, mask)
    torch.testing.assert_close(selected, hidden[[0, 1], [1, 3]])

