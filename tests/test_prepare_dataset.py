import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_dataset import prepare_dataset


class TestPrepareDataset(unittest.TestCase):
    def test_prepare_dataset_creates_clean_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            input_dir = root / "stories"
            input_dir.mkdir()

            story_dir = input_dir / "example"
            story_dir.mkdir()

            source_file = story_dir / "example.json"

            source_file.write_text(
                json.dumps(
                    {
                        "source_file": "example.pdf",
                        "page_count": 3,
                        "pages": [
                            {
                                "page_number": 1,
                                "text": (
                                    "Salam,     dünýä!\n\n"
                                    "ä ç ň ö ş ü ý ž"
                                ),
                                "character_count": 0,
                            },
                            {
                                "page_number": 2,
                                "text": "   \n\t ",
                                "character_count": 0,
                            },
                            {
                                "page_number": 3,
                                "text": "Soňky sahypa.",
                                "character_count": 0,
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            output_path = root / "processed" / "train.jsonl"

            record_count = prepare_dataset(
                input_dir=input_dir,
                output_path=output_path,
            )

            self.assertEqual(record_count, 2)
            self.assertTrue(output_path.exists())

            lines = output_path.read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(len(lines), 2)

            first_record = json.loads(lines[0])

            self.assertEqual(
                first_record["text"],
                "Salam, dünýä!\nä ç ň ö ş ü ý ž",
            )

            self.assertEqual(
                first_record["source"],
                "example/example.json",
            )

            self.assertEqual(
                first_record["page"],
                1,
            )

            second_record = json.loads(lines[1])

            self.assertEqual(
                second_record["text"],
                "Soňky sahypa.",
            )

            self.assertEqual(
                second_record["page"],
                3,
            )


if __name__ == "__main__":
    unittest.main()