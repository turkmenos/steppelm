"""Command-line entry point for training the SteppeLM tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .tokenizer import (
        DEFAULT_DATASET_PATH,
        DEFAULT_MIN_FREQUENCY,
        DEFAULT_TOKENIZER_PATH,
        DEFAULT_VOCAB_SIZE,
        SPECIAL_TOKENS,
        train_tokenizer,
    )
else:
    from tokenizer import (
        DEFAULT_DATASET_PATH,
        DEFAULT_MIN_FREQUENCY,
        DEFAULT_TOKENIZER_PATH,
        DEFAULT_VOCAB_SIZE,
        SPECIAL_TOKENS,
        train_tokenizer,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train the initial Turkmen BPE tokenizer."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Prepared JSONL training dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="Output tokenizer JSON path.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Target tokenizer vocabulary size.",
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
        help="Minimum pair frequency required for BPE merges.",
    )

    return parser.parse_args()


def main() -> None:
    """Train the tokenizer and report the result."""

    args = parse_args()

    tokenizer = train_tokenizer(
        dataset_path=args.dataset,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    print(f"Tokenizer saved to: {args.output}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    print("Special tokens:")

    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token}: {token_id}")


if __name__ == "__main__":
    main()