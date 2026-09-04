import json

import numpy as np

from src.prepare_data import deduplicate_lowest_row, select_rows, stable_id, write_jsonl


def test_stable_id_uses_quadrant_original_row_and_text_hash():
    identifier = stable_id("bench_eval", 17, "Question text")
    assert identifier.startswith("bench_eval__17__")
    assert len(identifier.rsplit("__", 1)[1]) == 12
    assert identifier == stable_id("bench_eval", 17, "Question text")
    assert identifier != stable_id("bench_eval", 18, "Question text")
    assert identifier != stable_id("bench_eval", 17, "Changed text")


def test_deduplication_keeps_lowest_original_row_after_manual_exclusions():
    rows = [
        {"prompt": "alpha"},
        {"prompt": "beta"},
        {"prompt": "alpha"},
        {"prompt": "gamma"},
    ]
    eligible, duplicate_rows = deduplicate_lowest_row(rows, "prompt", {1})
    assert eligible == [(0, rows[0]), (3, rows[3])]
    assert duplicate_rows == [2]


def test_seeded_selection_is_deterministic_and_sorted_by_original_row():
    rows = [(index, {"prompt": str(index)}) for index in range(20)]
    first = select_rows(rows, 7, np.random.default_rng(42))
    second = select_rows(rows, 7, np.random.default_rng(42))
    assert first == second
    assert [index for index, _ in first] == sorted(index for index, _ in first)


def test_jsonl_escapes_unicode_line_separators(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [{"text": "before\u2028after"}, {"text": "second"}]
    write_jsonl(path, records)
    raw_lines = path.read_text().splitlines()
    assert len(raw_lines) == 2
    assert [json.loads(line) for line in raw_lines] == records
