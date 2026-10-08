
import unittest

from prompt_parser import parse_search_prompt


class TestPromptParser(unittest.TestCase):

    def test_quality_request(self):
        intent = parse_search_prompt(
            "Find 5 good photos of me outside"
        )

        self.assertEqual(intent.query, "me outside")
        self.assertEqual(intent.limit, 5)
        self.assertTrue(intent.rerank)
        self.assertEqual(intent.selection, "none")

    def test_diversity_request(self):
        intent = parse_search_prompt(
            "Show me 10 different beach photos"
        )

        self.assertEqual(intent.query, "beach")
        self.assertEqual(intent.limit, 10)
        self.assertFalse(intent.rerank)
        self.assertEqual(intent.selection, "mmr")

    def test_deduplication_request(self):
        intent = parse_search_prompt(
            "Show 8 photos of me hiking without duplicates"
        )

        self.assertEqual(intent.query, "me hiking")
        self.assertEqual(intent.limit, 8)
        self.assertEqual(intent.selection, "deduplicate")

    def test_plain_query(self):
        intent = parse_search_prompt("outside")

        self.assertEqual(intent.query, "outside")
        self.assertEqual(intent.limit, 10)
        self.assertFalse(intent.rerank)
        self.assertEqual(intent.selection, "none")

    def test_empty_prompt(self):
        with self.assertRaises(ValueError):
            parse_search_prompt("   ")

    def test_zero_count(self):
        with self.assertRaises(ValueError):
            parse_search_prompt("Find 0 photos of me")

    def test_case_insensitive(self):
        intent = parse_search_prompt(
            "SHOW ME 4 BEST SUNSET PHOTOS"
        )

        self.assertEqual(intent.query.lower(), "sunset")
        self.assertEqual(intent.limit, 4)
        self.assertTrue(intent.rerank)

    def test_default_limit(self):
        intent = parse_search_prompt("Find photos of mountains")

        self.assertEqual(intent.limit, 10)


if __name__ == "__main__":
    unittest.main()
