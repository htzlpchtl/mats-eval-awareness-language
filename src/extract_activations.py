"""Extract final-token transformer-block activations (Stages 5 and 9)."""

import argparse
from pathlib import Path


FROZEN_ENGLISH_V2 = Path("data/selected/english_selected_v2.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=FROZEN_ENGLISH_V2)
    args = parser.parse_args()
    raise SystemExit(
        "Stage 5 is not implemented or authorised yet. "
        f"Frozen input would be: {args.dataset}"
    )


if __name__ == "__main__":
    main()

