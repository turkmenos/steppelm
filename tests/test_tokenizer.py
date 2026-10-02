import json
import tempfile
import unittest
from pathlib import Path

from src.tokenizer import (
    SPECIAL_TOKENS,
    load_tokenizer,
    train_tokenizer,
)


class TestTurkmenTokenizer(unittest.TestCase):
    def create_dataset(self, path: Path) -> None:
        records = [
            {
                "text": (
                    "Türkmenistanyň dili örän owadan."
                )
            },
            {
                "text": (
                    "Öýüň öňünde çynar agajy bar."
                )
            },
            {
                "text": (
                    "Şäheriň köçelerinde ýaşlar gezýär."
                )
            },
            {
                "text": (
                    "Ä Ç Ň Ö Ş Ü Ý Ž"
                )
            },
        ]

        with path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as dataset_file:
            for record in records:
                dataset_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def test_trains_and_saves_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            dataset_path = root / "train.jsonl"
            tokenizer_path = root / "tokenizer.json"

            self.create_dataset(dataset_path)

            tokenizer = train_tokenizer(
                dataset_path=dataset_path,
                output_path=tokenizer_path,
                vocab_size=512,
                min_frequency=1,
            )

            self.assertTrue(tokenizer_path.exists())
            self.assertGreater(
                tokenizer.get_vocab_size(),
                len(SPECIAL_TOKENS),
            )

    def test_special_tokens_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            dataset_path = root / "train.jsonl"
            tokenizer_path = root / "tokenizer.json"

            self.create_dataset(dataset_path)

            tokenizer = train_tokenizer(
                dataset_path=dataset_path,
                output_path=tokenizer_path,
                vocab_size=512,
                min_frequency=1,
            )

            token_ids = []

            for token in SPECIAL_TOKENS:
                token_id = tokenizer.token_to_id(token)

                self.assertIsNotNone(token_id)
                token_ids.append(token_id)

            self.assertEqual(
                len(token_ids),
                len(set(token_ids)),
            )

    def test_round_trip_preserves_turkmen_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            dataset_path = root / "train.jsonl"
            tokenizer_path = root / "tokenizer.json"

            self.create_dataset(dataset_path)

            tokenizer = train_tokenizer(
                dataset_path=dataset_path,
                output_path=tokenizer_path,
                vocab_size=512,
                min_frequency=1,
            )

            loaded_tokenizer = load_tokenizer(
                tokenizer_path
            )

            text = (
                "Äýşeňiň öýünde çynar bar: "
                "ö, ş, ü, ý, ž."
            )

            encoding = loaded_tokenizer.encode(text)

            self.assertGreater(
                len(encoding.ids),
                0,
            )

            decoded = loaded_tokenizer.decode(
                encoding.ids,
                skip_special_tokens=True,
            )

            self.assertEqual(decoded, text)

    def test_missing_dataset_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            dataset_path = root / "missing.jsonl"
            tokenizer_path = root / "tokenizer.json"

            with self.assertRaises(FileNotFoundError):
                train_tokenizer(
                    dataset_path=dataset_path,
                    output_path=tokenizer_path,
                    vocab_size=512,
                    min_frequency=1,
                )


if __name__ == "__main__":
    unittest.main()