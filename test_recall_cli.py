
import json
import sys
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import recall


class TestRecallCLI(unittest.TestCase):

    def run_cli(self, *arguments):
        """Run Recall's CLI without invoking the search models."""
        output = StringIO()

        with patch.object(
            sys,
            "argv",
            ["recall.py", "search", *arguments],
        ):
            with patch(
                "recall.search_photos",
                return_value=[],
            ) as mock_search:
                with patch("sys.stdout", output):
                    recall.main()

        return output.getvalue(), mock_search

    def test_interpreted_quality_prompt(self):
        output, mock_search = self.run_cli(
            "Find 5 good photos of me outside",
            "--interpret",
        )

        kwargs = mock_search.call_args.kwargs

        self.assertEqual(kwargs["query"], "me outside")
        self.assertEqual(kwargs["limit"], 5)
        self.assertTrue(kwargs["rerank"])
        self.assertEqual(kwargs["selection"], "none")
        self.assertIn("Quality-aware ranking", output)

    def test_interpreted_diversity_prompt(self):
        output, mock_search = self.run_cli(
            "Show me 10 different beach photos",
            "--interpret",
            "--json",
        )

        kwargs = mock_search.call_args.kwargs
        payload = json.loads(output)

        self.assertEqual(kwargs["query"], "beach")
        self.assertEqual(kwargs["limit"], 10)
        self.assertEqual(kwargs["selection"], "mmr")
        self.assertEqual(payload["selection"], "mmr")

    def test_explicit_limit_override(self):
        _, mock_search = self.run_cli(
            "Find 5 good photos of me outside",
            "--interpret",
            "--limit",
            "3",
        )

        self.assertEqual(
            mock_search.call_args.kwargs["limit"],
            3,
        )

    def test_explicit_selection_override(self):
        _, mock_search = self.run_cli(
            "Show me 10 different beach photos",
            "--interpret",
            "--deduplicate",
        )

        self.assertEqual(
            mock_search.call_args.kwargs["selection"],
            "deduplicate",
        )

    def test_plain_search_unchanged(self):
        _, mock_search = self.run_cli(
            "outside",
            "--limit",
            "3",
        )

        kwargs = mock_search.call_args.kwargs

        self.assertEqual(kwargs["query"], "outside")
        self.assertEqual(kwargs["limit"], 3)
        self.assertFalse(kwargs["rerank"])
        self.assertEqual(kwargs["selection"], "none")

    def test_json_output_contains_results(self):
        result = {
            "image_id": 39,
            "path": r"D:\Photos\photo1.JPG",
            "semantic_score": 0.25,
            "identity_score": 0.90,
            "quality_tier": 0,
            "quality_evidence": SimpleNamespace(
                blur_evidence=False,
                underexposure_evidence=False,
                overexposure_evidence=False,
            ),
        }

        with patch.object(
            sys,
            "argv",
            ["recall.py", "search", "outside", "--json"],
        ):
            with patch(
                "recall.search_photos",
                return_value=[result],
            ):
                with patch("sys.stdout", new_callable=StringIO) as output:
                    recall.main()

        payload = json.loads(output.getvalue())

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["image_id"], 39)
        self.assertEqual(payload["results"][0]["rank"], 1)


    @patch("recall.webbrowser.open")
    @patch("recall.render_gallery")
    @patch("recall.search_photos", return_value=[])
    def test_gallery_opens_browser(
        self,
        mock_search,
        mock_render,
        mock_browser,
    ):
        from pathlib import Path

        mock_render.return_value = Path(".recall/gallery.html")

        with patch.object(
            sys,
            "argv",
            [
                "recall.py",
                "search",
                "outside",
                "--gallery",
            ],
        ):
            with patch("sys.stdout", new_callable=StringIO):
                recall.main()

        mock_search.assert_called_once()
        mock_render.assert_called_once()

        self.assertEqual(
            mock_render.call_args.args[1],
            Path(".recall/gallery.html"),
        )

        mock_browser.assert_called_once_with(
            Path(".recall/gallery.html").resolve().as_uri()
        )

    @patch("recall.search_photos", return_value=[])
    def test_json_and_gallery_are_mutually_exclusive(
        self,
        mock_search,
    ):
        with patch.object(
            sys,
            "argv",
            [
                "recall.py",
                "search",
                "outside",
                "--json",
                "--gallery",
            ],
        ):
            with patch("sys.stderr", new_callable=StringIO):
                with self.assertRaises(SystemExit) as context:
                    recall.main()

        self.assertEqual(context.exception.code, 2)
        mock_search.assert_not_called()

if __name__ == "__main__":
    unittest.main()
