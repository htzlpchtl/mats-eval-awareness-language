from src.inspect_data import (
    duplicate_summary,
    is_non_latin_script_without_ascii_latin,
    obvious_malformed_reasons,
)


def test_duplicate_summary_counts_groups_rows_and_excess():
    assert duplicate_summary(["a", "a", "b", "c", "c", "c"]) == {
        "duplicate_groups": 2,
        "rows_in_duplicate_groups": 5,
        "excess_duplicate_rows": 3,
    }


def test_obvious_malformed_audit_is_conservative():
    assert obvious_malformed_reasons(None) == ["not_a_string"]
    assert obvious_malformed_reasons("  \n") == ["empty_or_whitespace"]
    assert obvious_malformed_reasons("hello") == []
    assert obvious_malformed_reasons("\ufffd broken") == [
        "unicode_replacement_character"
    ]


def test_non_latin_script_without_ascii_latin_flag():
    assert is_non_latin_script_without_ascii_latin("これは日本語です")
    assert is_non_latin_script_without_ascii_latin("這是中文")
    assert is_non_latin_script_without_ascii_latin("תענה לי באנגלית")
    assert not is_non_latin_script_without_ascii_latin("Answer in 日本語")
    assert not is_non_latin_script_without_ascii_latin("ordinary English prompt")
