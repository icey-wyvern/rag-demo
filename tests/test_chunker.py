import unittest

from api.chunker import make_chunks


class ChunkerTests(unittest.TestCase):
    def test_empty_input_returns_no_chunks(self):
        self.assertEqual(make_chunks([]), [])

    def test_groups_short_paragraphs_into_chunks(self):
        chunks = make_chunks(
            ["alpha beta gamma", "delta epsilon", "zeta eta theta"],
            max_tokens=50,
            overlap_tokens=0,
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("alpha beta gamma", chunks[0][1])

    def test_splits_long_paragraph(self):
        long_paragraph = " ".join(f"word{i}" for i in range(200))
        chunks = make_chunks([long_paragraph], max_tokens=20, overlap_tokens=0)
        self.assertGreater(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
