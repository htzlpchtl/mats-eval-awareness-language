"""Inspect the live Stage 1 dataset without selecting the frozen sample."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


DATASET_ID = "viliana-dev/eval-awareness-2x2"
QUADRANTS = {
    "bench_eval": {
        "path": "bench_eval/train-00000-of-00001.parquet",
        "candidate_text_field": "formatted_prompt",
    },
    "casual_eval": {
        "path": "casual_eval_mutual/train-00000-of-00001.parquet",
        "candidate_text_field": "first_turn",
    },
    "bench_deploy": {
        "path": "bench_deploy_rewritten/train-00000-of-00001.parquet",
        "candidate_text_field": "rewritten",
    },
    "casual_deploy": {
        "path": "casual_deploy_matched/train-00000-of-00001.parquet",
        "candidate_text_field": "first_turn",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/stage1_dataset_inspection.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("results/metrics/CHECKPOINT_1.md"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-per-quadrant", type=int, default=10)
    return parser.parse_args()


def empty_count(values: list[Any]) -> int:
    return sum(
        value is None or (isinstance(value, str) and not value.strip())
        for value in values
    )


def duplicate_summary(texts: list[str]) -> dict[str, int]:
    counts = Counter(texts)
    duplicate_groups = [count for count in counts.values() if count > 1]
    return {
        "duplicate_groups": len(duplicate_groups),
        "rows_in_duplicate_groups": sum(duplicate_groups),
        "excess_duplicate_rows": sum(count - 1 for count in duplicate_groups),
    }


def obvious_malformed_reasons(value: Any) -> list[str]:
    """Return conservative structural problems, without judging prompt quality."""

    if not isinstance(value, str):
        return ["not_a_string"]
    if not value.strip():
        return ["empty_or_whitespace"]
    reasons = []
    if "\x00" in value:
        reasons.append("embedded_nul")
    if "\ufffd" in value:
        reasons.append("unicode_replacement_character")
    if not any(character.isalnum() for character in value):
        reasons.append("no_alphanumeric_characters")
    return reasons


def is_non_latin_script_without_ascii_latin(value: Any) -> bool:
    """Flag clearly non-English-script text for human review."""

    if not isinstance(value, str):
        return False
    has_latin = bool(re.search(r"[A-Za-z]", value))
    return any(character.isalpha() for character in value) and not has_latin


def inspect_quadrant(
    spec: dict[str, str],
    *,
    revision: str,
    rng: np.random.Generator,
    examples_per_quadrant: int,
) -> tuple[dict[str, Any], list[Any]]:
    local_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=spec["path"],
        revision=revision,
    )
    table = pq.read_table(local_path)
    rows = table.to_pylist()
    field_values = {name: table[name].to_pylist() for name in table.column_names}
    text_field = spec["candidate_text_field"]
    raw_texts = field_values[text_field]
    valid_texts = [
        text for text in raw_texts if isinstance(text, str) and text.strip()
    ]
    malformed_text_rows = [
        {"row_index": index, "reasons": reasons, "fields": rows[index]}
        for index, text in enumerate(raw_texts)
        if (reasons := obvious_malformed_reasons(text))
    ]
    non_latin_script_rows = [
        {
            "row_index": index,
            "reason": "contains alphabetic characters but no ASCII Latin letters",
            "fields": rows[index],
        }
        for index, text in enumerate(raw_texts)
        if is_non_latin_script_without_ascii_latin(text)
    ]
    sample_size = min(examples_per_quadrant, len(rows))
    sample_indices = sorted(
        int(index)
        for index in rng.choice(len(rows), size=sample_size, replace=False)
    )

    identifier_fields = [
        name
        for name in table.column_names
        if name == "id" or name.endswith("_id")
    ]
    identifiers = {
        name: {
            "empty": empty_count(field_values[name]),
            "unique_nonempty": len(
                {
                    value
                    for value in field_values[name]
                    if value is not None
                    and (not isinstance(value, str) or value.strip())
                }
            ),
        }
        for name in identifier_fields
    }

    report = {
        "repository_path": spec["path"],
        "row_count": len(rows),
        "arrow_schema": str(table.schema),
        "columns": {
            name: {
                "arrow_type": str(table.schema.field(name).type),
                "python_types": dict(
                    sorted(Counter(type(value).__name__ for value in values).items())
                ),
                "missing_or_empty": empty_count(values),
            }
            for name, values in field_values.items()
        },
        "identifier_fields": identifiers,
        "proposed_model_facing_field": text_field,
        "proposal_status": "requires human approval",
        "candidate_prompt_integrity": {
            "obviously_malformed_row_count": len(malformed_text_rows),
            "obviously_malformed_rows": malformed_text_rows,
            "non_latin_script_row_count": len(non_latin_script_rows),
            "non_latin_script_rows": non_latin_script_rows,
            **duplicate_summary(valid_texts),
        },
        "deterministic_examples": [
            {"row_index": index, "fields": rows[index]} for index in sample_indices
        ],
    }
    return report, raw_texts


def render_markdown(output: dict[str, Any]) -> str:
    """Render the complete human-review checkpoint from the JSON object."""

    lines = [
        "# Checkpoint 1: live dataset inspection",
        "",
        f"- Dataset: `{output['dataset_id']}`",
        f"- Revision: `{output['dataset_revision']}`",
        f"- Seed: {output['seed']}",
        f"- Storage note: {output['storage_finding']}",
        "- No rows were sampled, frozen, excluded, or modified.",
        "",
        "## Proposed stable ID",
        "",
        f"`{output['proposed_stable_id']['format']}`",
        "",
        output["proposed_stable_id"]["rationale"],
    ]
    for quadrant, report in output["quadrants"].items():
        integrity = report["candidate_prompt_integrity"]
        lines.extend(
            [
                "",
                f"## {quadrant}",
                "",
                f"- Rows: {report['row_count']}",
                f"- Proposed model-facing field: `{report['proposed_model_facing_field']}` (requires approval)",
                f"- Identifier audit: `{json.dumps(report['identifier_fields'], ensure_ascii=False)}`",
                f"- Exact-prompt duplicates: {integrity['duplicate_groups']} groups, "
                f"{integrity['rows_in_duplicate_groups']} rows, "
                f"{integrity['excess_duplicate_rows']} excess rows",
                f"- Obviously malformed: {integrity['obviously_malformed_row_count']}",
                f"- Non-Latin-script flags: {integrity['non_latin_script_row_count']}",
                "",
                "### Exact schema",
                "",
                "```text",
                report["arrow_schema"],
                "```",
                "",
                "### Field missing/empty counts",
                "",
                "| Field | Arrow type | Missing/empty |",
                "|---|---:|---:|",
            ]
        )
        for field, details in report["columns"].items():
            lines.append(
                f"| `{field}` | `{details['arrow_type']}` | {details['missing_or_empty']} |"
            )
        lines.extend(["", "### Ten deterministic raw examples", ""])
        for example in report["deterministic_examples"]:
            lines.extend(
                [
                    f"Row {example['row_index']}:",
                    "",
                    "```json",
                    json.dumps(example["fields"], indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )

    lines.extend(["## Every flagged row", ""])
    flagged_count = 0
    for quadrant, report in output["quadrants"].items():
        integrity = report["candidate_prompt_integrity"]
        for category in ("obviously_malformed_rows", "non_latin_script_rows"):
            for flagged in integrity[category]:
                flagged_count += 1
                lines.extend(
                    [
                        f"### {quadrant} row {flagged['row_index']}",
                        "",
                        f"Reason: {flagged.get('reason', flagged.get('reasons'))}",
                        "",
                        "```json",
                        json.dumps(flagged["fields"], indent=2, ensure_ascii=False),
                        "```",
                        "",
                    ]
                )
    if not flagged_count:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Cross-quadrant exact candidate-prompt duplicates",
            "",
            f"Groups: {output['cross_quadrant_candidate_prompt_duplicates']['duplicate_groups']}",
            "",
            "```json",
            json.dumps(
                output["cross_quadrant_candidate_prompt_duplicates"]["groups"],
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset_info = HfApi().dataset_info(DATASET_ID, files_metadata=True)
    revision = dataset_info.sha
    available_files = sorted(sibling.rfilename for sibling in dataset_info.siblings)
    rng = np.random.default_rng(args.seed)

    quadrant_reports: dict[str, Any] = {}
    texts_by_quadrant: dict[str, list[Any]] = {}
    for quadrant, spec in QUADRANTS.items():
        report, texts = inspect_quadrant(
            spec,
            revision=revision,
            rng=rng,
            examples_per_quadrant=args.examples_per_quadrant,
        )
        quadrant_reports[quadrant] = report
        texts_by_quadrant[quadrant] = texts

    locations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for quadrant, texts in texts_by_quadrant.items():
        for row_index, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                locations[text].append(
                    {"quadrant": quadrant, "row_index": row_index}
                )
    cross_quadrant_duplicates = [
        {"text": text, "locations": found}
        for text, found in locations.items()
        if len({item["quadrant"] for item in found}) > 1
    ]

    output = {
        "stage": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_revision": revision,
        "storage_finding": (
            "The installed datasets API reports only a default config despite the "
            "dataset card declaring the named configs. The intended quadrants are "
            "separate Parquet subdirectories with incompatible schemas, so this "
            "inspection reads the four files directly at the pinned revision."
        ),
        "available_repository_files": available_files,
        "seed": args.seed,
        "examples_per_quadrant": args.examples_per_quadrant,
        "proposed_stable_id": {
            "format": "{quadrant}:{original_row_index}:{sha256(model_facing_text)[:12]}",
            "status": "requires human approval",
            "rationale": (
                "Quadrant and original row index preserve provenance and uniqueness; "
                "the short text hash detects accidental field or content changes. The "
                "source id is retained as metadata but cannot be the primary key because "
                "693 bench_eval source ids are empty."
            ),
        },
        "quadrants": quadrant_reports,
        "cross_quadrant_candidate_prompt_duplicates": {
            "duplicate_groups": len(cross_quadrant_duplicates),
            "groups": cross_quadrant_duplicates,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(output))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
