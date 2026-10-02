"""Utilities for reading Turkmen story data from tm-data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STORIES_DIR = (
    PROJECT_ROOT / "data" / "tm-data" / "stories"
)


class StoryPage(TypedDict):
    """Normalized representation of a source story page."""

    source: str
    page_number: int | None
    text: str


def iter_story_pages(
    stories_dir: Path = DEFAULT_STORIES_DIR,
) -> Iterator[StoryPage]:
    """Yield page records from all story JSON files.

    The source files are read explicitly as UTF-8. The function expects
    the page-level JSON structure documented by tm-data:

        {
            "pages": [
                {
                    "page_number": 1,
                    "text": "..."
                }
            ]
        }

    Args:
        stories_dir: Directory containing tm-data story JSON files.

    Yields:
        StoryPage records containing source path, page number, and text.

    Raises:
        FileNotFoundError: If the stories directory does not exist or
            contains no JSON files.
        ValueError: If a JSON document does not contain a valid ``pages``
            list or contains an invalid page object.
        UnicodeDecodeError: If a source file is not valid UTF-8.
        json.JSONDecodeError: If a source file is not valid JSON.
    """
    if not stories_dir.is_dir():
        raise FileNotFoundError(
            f"Stories directory does not exist: {stories_dir}"
        )

    json_paths = sorted(stories_dir.rglob("*.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"No story JSON files found under {stories_dir}"
        )

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as input_file:
            document = json.load(input_file)

        if not isinstance(document, dict):
            raise ValueError(
                f"Expected a JSON object in {json_path}"
            )

        pages = document.get("pages")

        if not isinstance(pages, list):
            raise ValueError(
                f"Expected a 'pages' list in {json_path}"
            )

        source = json_path.relative_to(stories_dir).as_posix()

        for page in pages:
            if not isinstance(page, dict):
                raise ValueError(
                    f"Expected every page in {json_path} to be an object"
                )

            raw_text = page.get("text")

            if not isinstance(raw_text, str):
                continue

            yield {
                "source": source,
                "page_number": page.get("page_number"),
                "text": raw_text,
            }


def iter_story_texts(
    stories_dir: Path = DEFAULT_STORIES_DIR,
) -> Iterator[str]:
    """Yield non-empty raw page text from tm-data."""
    for page in iter_story_pages(stories_dir):
        if page["text"].strip():
            yield page["text"]


if __name__ == "__main__":
    pages = list(iter_story_pages())

    character_count = sum(
        len(page["text"])
        for page in pages
    )

    print(
        f"Loaded {len(pages)} pages "
        f"({character_count} characters)."
    )