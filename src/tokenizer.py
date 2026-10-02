"""BPE tokenizer utilities for the SteppeLM project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers import processors, trainers

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train.jsonl"
)

DEFAULT_TOKENIZER_PATH = (
    PROJECT_ROOT / "checkpoints" / "tokenizer.json"
)

SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<mask>",
)

DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MIN_FREQUENCY = 2


def iter_training_texts(dataset_path: Path) -> Iterator[str]:
    """Yield training text from the prepared JSONL dataset.

    Args:
        dataset_path: Path to the prepared JSONL dataset.

    Yields:
        Non-empty text values from each JSONL record.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If a JSONL record is malformed.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Training dataset does not exist: {dataset_path}"
        )

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"in {dataset_path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} "
                    f"in {dataset_path}"
                )

            text = record.get("text")

            if not isinstance(text, str):
                raise ValueError(
                    f"Missing string 'text' field on line {line_number} "
                    f"in {dataset_path}"
                )

            text = text.strip()

            if text:
                yield text


def create_tokenizer(
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
) -> Tokenizer:
    """Create an untrained Turkmen BPE tokenizer."""

    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(
            "vocab_size must be greater than the number "
            "of special tokens"
        )

    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>",
        )
    )

    # Keep canonically equivalent Unicode representations consistent.
    tokenizer.normalizer = normalizers.NFC()

    # Byte-level tokenization allows every UTF-8 byte to be represented,
    # including all Turkmen-specific characters.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def train_tokenizer(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path = DEFAULT_TOKENIZER_PATH,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
) -> Tokenizer:
    """Train and save the SteppeLM Turkmen BPE tokenizer."""

    tokenizer = create_tokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(
        iter_training_texts(dataset_path),
        trainer=trainer,
    )

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    if bos_id is None or eos_id is None:
        raise RuntimeError(
            "Required special tokens were not created"
        )

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
        ],
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(output_path))

    return tokenizer


def load_tokenizer(
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
) -> Tokenizer:
    """Load a saved SteppeLM tokenizer."""

    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer file does not exist: {tokenizer_path}"
        )

    return Tokenizer.from_file(str(tokenizer_path))