
import unittest
from unittest.mock import patch

from search_api import search_photos


class TestSearchAPI(unittest.TestCase):

    def test_empty_query(self):
        with self.assertRaises(ValueError):
            search_photos("   ")

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            search_photos("outside", limit=0)

    def test_invalid_selection(self):
        with self.assertRaises(ValueError):
            search_photos(
                "outside",
                selection="unknown",
            )

    def test_invalid_diversity_weight(self):
        with self.assertRaises(ValueError):
            search_photos(
                "outside",
                diversity_weight=1.5,
            )

    @patch("search_api.select_mmr")
    @patch("search_api.load_candidate_embeddings")
    @patch("search_api.search_me_with_quality")
    @patch("search_api.connect")
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mmr_pipeline(
        self,
        mock_is_file,
        mock_connect,
        mock_search,
        mock_embeddings,
        mock_mmr,
    ):
        connection = mock_connect.return_value

        candidates = [
            {"image_id": 1, "quality_tier": 0},
            {"image_id": 2, "quality_tier": 1},
        ]

        mock_search.return_value = candidates
        mock_embeddings.return_value = {}
        mock_mmr.return_value = [candidates[0]]

        results = search_photos(
            "outside",
            limit=1,
            pool_size=30,
            rerank=True,
            selection="mmr",
            diversity_weight=0.4,
        )

        self.assertEqual(results, [candidates[0]])

        mock_search.assert_called_once_with(
            connection,
            "outside",
            unittest.mock.ANY,
            identity_limit=100,
            candidate_limit=30,
            rerank=True,
        )

        mock_mmr.assert_called_once_with(
            candidates,
            {},
            limit=1,
            diversity_weight=0.4,
        )

        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
