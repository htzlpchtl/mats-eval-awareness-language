"""Run lightweight structural QA over held-out Spanish/Japanese translations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.extract_activations import (
    FROZEN_ENGLISH_V2,
    FROZEN_ENGLISH_V2_SHA256,
    QUADRANTS,
    sha256_file,
)
from src.translate_prompts import frozen_test_rows, validate_output_rows

TRANSLATION_PATHS = {
    "spanish": Path("data/translated/spanish_test_v2.jsonl"),
    "japanese": Path("data/translated/japanese_test_v2.jsonl"),
}
NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?:%|st|nd|rd|th)?", re.I)
CHOICE_RE = re.compile(r"(?<![A-Za-z0-9])(?:\([A-H]\)|[A-H][.)])(?=\s|$)")
URL_RE = re.compile(r"https?://[^\s)>\]}]+")
KANA_RE = re.compile(r"[\u3040-\u30ff]")
PREAMBLE_RE = re.compile(
    r"^(?:translation|translated text|traducci[o\u00f3]n|texto traducido|\u7ffb\u8a33|\u8a33\u6587)\s*[:\uff1a]",
    re.I,
)
ANSWERED_RE = re.compile(
    r"(?:Answer|Respuesta|\u7b54\u3048|\u56de\u7b54)\s*[:\uff1a]\s*(?!$)(?:\([A-H]\)|[A-H][.)]|\S{2,})",
    re.I,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def structural_qa(english: str, translated: str, language: str) -> dict[str, Any]:
    english_numbers = NUMBER_RE.findall(english)
    translated_numbers = NUMBER_RE.findall(translated)
    english_choices = CHOICE_RE.findall(english)
    translated_choices = CHOICE_RE.findall(translated)
    english_urls = URL_RE.findall(english)
    translated_urls = URL_RE.findall(translated)
    ratio = len(translated) / len(english) if english else float("inf")
    lower_limit = 0.20 if language == "japanese" else 0.45
    upper_limit = 2.25
    flags: list[str] = []
    if not translated.strip():
        flags.append("empty_translation")
    if translated.strip() == english.strip():
        flags.append("identical_to_english")
    if language == "japanese" and translated.strip() and not KANA_RE.search(translated):
        flags.append("target_script_absent")
    if not lower_limit <= ratio <= upper_limit:
        flags.append("extreme_character_length_ratio")
    if english_choices != translated_choices:
        flags.append("answer_choice_labels_changed")
    if english_numbers != translated_numbers:
        flags.append("numbers_changed")
    if english_urls != translated_urls:
        flags.append("urls_changed")
    if english.count("`") != translated.count("`"):
        flags.append("backtick_count_changed")
    if PREAMBLE_RE.search(translated.strip()):
        flags.append("translator_preamble")
    english_answered = bool(ANSWERED_RE.search(english.strip()))
    if not english_answered and ANSWERED_RE.search(translated.strip()):
        flags.append("possible_answer_generation")
    return {
        "english_characters": len(english),
        "translated_characters": len(translated),
        "character_length_ratio": ratio,
        "english_newlines": english.count("\n"),
        "translated_newlines": translated.count("\n"),
        "english_choice_count": len(english_choices),
        "translated_choice_count": len(translated_choices),
        "english_number_count": len(english_numbers),
        "translated_number_count": len(translated_numbers),
        "english_url_count": len(english_urls),
        "translated_url_count": len(translated_urls),
        "flags": flags,
        "qa_pass": not flags,
    }


def qa_rows(translations: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for language, rows in translations.items():
        for row in rows:
            checks = structural_qa(
                row["original_english_text"], row["model_facing_text"], language
            )
            output.append(
                {
                    "stable_id": row["stable_id"],
                    "manifest_row_index": row["manifest_row_index"],
                    "language": language,
                    "quadrant": row["quadrant"],
                    "split": row["split"],
                    **{key: value for key, value in checks.items() if key != "flags"},
                    "flags": ";".join(checks["flags"]),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fixed_manual_ids(
    sources: list[tuple[int, dict[str, Any]]],
    seed: int = 42,
    per_quadrant: int = 2,
) -> list[str]:
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for _index, row in sources:
        grouped[row["quadrant"]].append(row["stable_id"])
    selected = []
    for quadrant in QUADRANTS:
        selected.extend(rng.sample(grouped[quadrant], per_quadrant))
    return selected


def indented(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


def write_manual_sample(
    path: Path,
    translations: dict[str, list[dict[str, Any]]],
    sample_ids: list[str],
) -> None:
    lookups = {
        language: {row["stable_id"]: row for row in rows}
        for language, rows in translations.items()
    }
    lines = [
        "# Minimal multilingual manual-audit sample",
        "",
        "Frozen seed 42; two held-out IDs per quadrant; the same IDs in both languages.",
        "Structural presentation was inspected, but Japanese semantic nuance was not human-fluency validated.",
        "",
    ]
    for stable_id in sample_ids:
        spanish = lookups["spanish"][stable_id]
        japanese = lookups["japanese"][stable_id]
        lines.extend(
            [
                f"## {stable_id} ({spanish['quadrant']})",
                "",
                "English:",
                "",
                indented(spanish["original_english_text"]),
                "",
                "Spanish:",
                "",
                indented(spanish["model_facing_text"]),
                "",
                "Japanese:",
                "",
                indented(japanese["model_facing_text"]),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=FROZEN_ENGLISH_V2)
    parser.add_argument("--dataset-sha256", default=FROZEN_ENGLISH_V2_SHA256)
    parser.add_argument(
        "--qa-csv", type=Path, default=Path("results/metrics/stage9_translation_qa.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/metrics/stage9_translation_qa.json")
    )
    parser.add_argument(
        "--manual-sample",
        type=Path,
        default=Path("results/metrics/STAGE9_TRANSLATION_SAMPLE.md"),
    )
    args = parser.parse_args()
    sources = frozen_test_rows(args.dataset, args.dataset_sha256)
    translations: dict[str, list[dict[str, Any]]] = {}
    for language, path in TRANSLATION_PATHS.items():
        rows = load_jsonl(path)
        validate_output_rows(rows, sources, language)
        translations[language] = rows

    rows = qa_rows(translations)
    write_csv(args.qa_csv, rows)
    sample_ids = fixed_manual_ids(sources)
    write_manual_sample(args.manual_sample, translations, sample_ids)
    flag_counts: dict[str, Counter[str]] = {}
    ratio_summary = {}
    for language in TRANSLATION_PATHS:
        selected = [row for row in rows if row["language"] == language]
        counter: Counter[str] = Counter()
        for row in selected:
            counter.update(flag for flag in row["flags"].split(";") if flag)
        flag_counts[language] = counter
        ratios = np.asarray([float(row["character_length_ratio"]) for row in selected])
        ratio_summary[language] = {
            "min": float(ratios.min()),
            "median": float(np.median(ratios)),
            "max": float(ratios.max()),
        }
    report = {
        "stage": "minimal held-out translation QA",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256_file(args.dataset),
        "rows_checked": len(rows),
        "rows_per_language": 360,
        "metadata_id_order_split_preservation": True,
        "translation_files": {
            language: {"path": str(path), "sha256": sha256_file(path)}
            for language, path in TRANSLATION_PATHS.items()
        },
        "flag_counts": {
            language: dict(counter) for language, counter in flag_counts.items()
        },
        "rows_with_any_flag": {
            language: sum(
                bool(row["flags"]) for row in rows if row["language"] == language
            )
            for language in TRANSLATION_PATHS
        },
        "character_length_ratio": ratio_summary,
        "manual_sample_ids": sample_ids,
        "manual_sample_rows": len(sample_ids) * len(TRANSLATION_PATHS),
        "manual_review_scope": (
            "Structural presentation only; Japanese semantic nuance was not "
            "human-fluency validated."
        ),
        "qa_policy": "Flag only; no rows removed or modified.",
        "qa_csv": str(args.qa_csv),
        "manual_sample": str(args.manual_sample),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
