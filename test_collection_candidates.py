
import unittest
from unittest.mock import patch

from collection_planner import create_collection_plan
from collection_candidates import retrieve_collection_candidates


class TestCollectionCandidates(unittest.TestCase):

    @patch("collection_candidates.search_photos")
    def test_retrieves_all_six_roles(self, mock_search):
        mock_search.return_value = []

        plan = create_collection_plan("Hinge")

        results = retrieve_collection_candidates(plan)

        self.assertEqual(results, [])
        self.assertEqual(mock_search.call_count, 6)

    @patch("collection_candidates.search_photos")
    def test_merges_same_image_across_roles(self, mock_search):
        mock_search.return_value = [
            {
                "image_id": 42,
                "path": "photo.jpg",
                "semantic_score": 0.75,
            }
        ]

        plan = create_collection_plan("Hinge")

        candidates = retrieve_collection_candidates(plan)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].image_id, 42)
        self.assertEqual(len(candidates[0].role_scores), 6)
        self.assertEqual(len(candidates[0].role_ranks), 6)

    @patch("collection_candidates.search_photos")
    def test_preserves_role_ranking(self, mock_search):
        mock_search.return_value = [
            {
                "image_id": 10,
                "path": "first.jpg",
                "semantic_score": 0.9,
            },
            {
                "image_id": 20,
                "path": "second.jpg",
                "semantic_score": 0.7,
            },
        ]

        plan = create_collection_plan("Hinge")

        candidates = retrieve_collection_candidates(plan)

        first = next(c for c in candidates if c.image_id == 10)

        self.assertEqual(
            first.role_ranks["main_portrait"],
            1,
        )

    @patch("collection_candidates.search_photos")
    def test_invalid_limits(self, mock_search):
        plan = create_collection_plan("Hinge")

        with self.assertRaises(ValueError):
            retrieve_collection_candidates(
                plan,
                per_role_limit=0,
            )

        with self.assertRaises(ValueError):
            retrieve_collection_candidates(
                plan,
                identity_limit=0,
            )

        mock_search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
