import numpy as np

from src.evaluate_multilingual_transfer import expected_calibration_error
from src.translation_qa import fixed_manual_ids, structural_qa
from src.translate_prompts import (
    LANGUAGES,
    TRANSLATION_MODEL,
    frozen_test_rows,
    translation_row,
    validate_output_rows,
)


def test_frozen_multilingual_selection_is_test_only_and_ordered():
    sources = frozen_test_rows()
    assert len(sources) == 360
    assert [index for index, _row in sources] == sorted(index for index, _row in sources)
    assert len({row["stable_id"] for _index, row in sources}) == 360
    assert all(row["split"] == "test" and row["cv_fold"] is None for _index, row in sources)
    assert set(LANGUAGES) == {"spanish", "japanese"}
    assert TRANSLATION_MODEL == "deepseek/deepseek-v4-flash-0731"


def test_translation_metadata_validation_preserves_all_held_out_ids():
    sources = frozen_test_rows()
    translated = [
        translation_row(row, index, "spanish", "Traduccion: " + row["model_facing_text"], "model", "id", None)
        for index, row in sources
    ]
    validate_output_rows(translated, sources, "spanish")
    assert [row["stable_id"] for row in translated] == [row["stable_id"] for _index, row in sources]


def test_structural_qa_flags_changed_choices_and_numbers():
    good = structural_qa("Question 12\nA) one B) two", "Pregunta 12\nA) uno B) dos", "spanish")
    assert good["qa_pass"]
    bad = structural_qa("Question 12\nA) one B) two", "Pregunta 13\nA) uno", "spanish")
    assert "numbers_changed" in bad["flags"]
    assert "answer_choice_labels_changed" in bad["flags"]


def test_manual_sample_is_same_two_ids_per_quadrant():
    selected = fixed_manual_ids(frozen_test_rows())
    assert len(selected) == 8
    assert len(set(selected)) == 8
    assert selected == fixed_manual_ids(frozen_test_rows())


def test_calibration_error_is_zero_for_perfect_binary_probabilities():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(labels, probabilities) == 0.0
