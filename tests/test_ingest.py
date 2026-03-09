import unittest

from api.ingest import _build_pages_and_texts


class IngestHelpersTests(unittest.TestCase):
    def test_build_pages_and_texts_preserves_page_numbers(self):
        pages_and_texts = _build_pages_and_texts(
            [
                {"page": 2, "text": "A" * 40},
                {"page": 2, "text": "B" * 40},
                {"page": 4, "text": "C" * 40},
            ]
        )
        self.assertEqual(pages_and_texts[0][0], 2)
        self.assertEqual(pages_and_texts[-1][0], 4)
        self.assertTrue(all(text for _, text in pages_and_texts))


if __name__ == "__main__":
    unittest.main()
