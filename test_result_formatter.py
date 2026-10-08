
import json
import unittest
from types import SimpleNamespace

from result_formatter import format_result, format_results


class TestResultFormatter(unittest.TestCase):

    def setUp(self):
        self.results = [
            {
                "image_id": 39,
                "path": r"D:\Photos\photo1.JPG",
                "semantic_score": 0.2457,
                "identity_score": 0.3430,
                "quality_tier": 0,
                "quality_evidence": SimpleNamespace(
                    blur_evidence=False,
                    underexposure_evidence=False,
                    overexposure_evidence=False,
                ),
            },
            {
                "image_id": 2,
                "path": r"D:\Photos\photo2.JPG",
                "semantic_score": 0.2385,
                "identity_score": 0.5223,
                "quality_tier": 2,
                "quality_evidence": SimpleNamespace(
                    blur_evidence=True,
                    underexposure_evidence=True,
                    overexposure_evidence=False,
                ),
            },
        ]

    def test_format_result(self):
        formatted = format_result(self.results[0], rank=1)

        self.assertEqual(formatted["rank"], 1)
        self.assertEqual(formatted["image_id"], 39)
        self.assertEqual(formatted["quality_tier"], 0)
        self.assertEqual(formatted["defects"], [])

    def test_defect_serialization(self):
        formatted = format_result(self.results[1], rank=2)

        self.assertEqual(
            formatted["defects"],
            ["blur", "underexposed"],
        )

    def test_result_count_and_ranking(self):
        payload = format_results(
            self.results,
            query="outside",
            rerank=True,
            selection="mmr",
        )

        self.assertEqual(payload["query"], "outside")
        self.assertEqual(payload["ranking"], "quality-aware")
        self.assertEqual(payload["selection"], "mmr")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            [item["rank"] for item in payload["results"]],
            [1, 2],
        )

    def test_json_serializable(self):
        payload = format_results(
            self.results,
            query="outside",
            rerank=False,
            selection="none",
        )

        serialized = json.dumps(payload)
        decoded = json.loads(serialized)

        self.assertEqual(decoded["count"], 2)
        self.assertEqual(decoded["ranking"], "semantic-only")

    def test_empty_results(self):
        payload = format_results(
            [],
            query="outside",
            rerank=False,
            selection="none",
        )

        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["results"], [])


if __name__ == "__main__":
    unittest.main()
