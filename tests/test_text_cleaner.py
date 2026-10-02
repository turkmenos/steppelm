import unittest

from src.text_cleaner import clean_text


class TestCleanText(unittest.TestCase):
    def test_removes_empty_lines(self):
        text = "Birinji setir\n\n   \nIkinji setir"

        self.assertEqual(
            clean_text(text),
            "Birinji setir\nIkinji setir",
        )

    def test_normalizes_repeated_horizontal_whitespace(self):
        text = "Salam,\t\t   dünýä!"

        self.assertEqual(
            clean_text(text),
            "Salam, dünýä!",
        )

    def test_normalizes_non_breaking_spaces(self):
        text = "Salam,\u00a0\u00a0dünýä!"

        self.assertEqual(
            clean_text(text),
            "Salam, dünýä!",
        )

    def test_preserves_turkmen_characters(self):
        text = "ä ç ň ö ş ü ý ž Ä Ç Ň Ö Ş Ü Ý Ž"

        self.assertEqual(
            clean_text(text),
            text,
        )

    def test_normalizes_unicode_to_nfc(self):
        text = "a\u0308"

        self.assertEqual(
            clean_text(text),
            "ä",
        )

    def test_preserves_meaningful_line_breaks(self):
        text = "Birinji setir\nIkinji setir"

        self.assertEqual(
            clean_text(text),
            "Birinji setir\nIkinji setir",
        )

    def test_rejects_non_string_input(self):
        with self.assertRaises(TypeError):
            clean_text(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()