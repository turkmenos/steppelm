"""Build a cleaned JSONL training dataset from tm-data stories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .data_loader import DEFAULT_STORIES_DIR, iter_story_pages
    from .text_cleaner import clean_text
else:
    from data_loader import DEFAULT_STORIES_DIR, iter_story_pages
    from text_cleaner import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train.jsonl"
)


def prepare_dataset(
    input_dir: Path,
    output_path: Path,
) -> int:
    """Clean all story pages and write them as JSONL records.

    Each output record contains:

    - ``text``: cleaned Turkmen text;
    - ``source``: source JSON path relative to the stories directory;
    - ``page``: original source page number.

    Args:
        input_dir: Directory containing tm-data story JSON files.
        output_path: Destination JSONL file.

    Returns:
        Number of records written.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record_count = 0

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        for page in iter_story_pages(input_dir):
            text = clean_text(page["text"])

            if not text:
                continue

            record = {
                "text": text,
                "source": page["source"],
                "page": page["page_number"],
            }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            record_count += 1

    return record_count


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare cleaned Turkmen story data "
            "from tm-data as JSONL."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_STORIES_DIR,
        help="Directory containing tm-data story JSON files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the generated JSONL dataset.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the dataset preparation pipeline."""
    args = parse_args()

    record_count = prepare_dataset(
        input_dir=args.input_dir,
        output_path=args.output,
    )

    print(
        f"Wrote {record_count} records "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()