"""Text normalization utilities for Turkmen training data."""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")


def clean_text(text: str) -> str:
    """Normalize and clean extracted Turkmen text.

    The function:
    - requires a string input;
    - normalizes Unicode to NFC;
    - preserves Turkmen-specific characters;
    - converts repeated horizontal whitespace to one space;
    - removes leading/trailing whitespace from each line;
    - removes empty lines;
    - preserves meaningful line boundaries.

    Args:
        text: Raw text extracted from the source dataset.

    Returns:
        Cleaned UTF-8-compatible Unicode text.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # NFC keeps canonically equivalent Unicode representations consistent
    # without changing the actual Turkmen letters.
    normalized_text = unicodedata.normalize("NFC", text)

    cleaned_lines: list[str] = []

    for line in normalized_text.splitlines():
        # PDF extraction can contain non-breaking spaces. Treat them as
        # ordinary spaces before collapsing repeated whitespace.
        line = line.replace("\u00a0", " ")

        # Normalize horizontal whitespace without touching meaningful
        # line boundaries.
        line = _WHITESPACE_RE.sub(" ", line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)