
import unittest

from collection_candidates import CollectionCandidate
from collection_planner import create_collection_plan
from photo_roles import (
    best_candidate_role,
    classify_candidate_roles,
)


class TestPhotoRoles(unittest.TestCase):

    def setUp(self):
        self.plan = create_collection_plan("Hinge")

    def test_returns_sorted_role_matches(self):
        candidate = CollectionCandidate(
            image_id=42,
            path="photo.jpg",
            role_scores={
                "group": 0.70,
                "activity": 0.85,
                "main_portrait": 0.60,
            },
            role_ranks={
                "group": 2,
                "activity": 1,
                "main_portrait": 3,
            },
        )

        matches = classify_candidate_roles(
            candidate,
            self.plan,
        )

        self.assertEqual(
            [match.role for match in matches],
            ["activity", "group", "main_portrait"],
        )

    def test_best_role(self):
        candidate = CollectionCandidate(
            image_id=10,
            path="photo.jpg",
            role_scores={
                "group": 0.75,
                "personality": 0.90,
            },
        )

        match = best_candidate_role(candidate, self.plan)

        self.assertEqual(match.role, "personality")
        self.assertAlmostEqual(match.score, 0.90)

    def test_unknown_roles_are_ignored(self):
        candidate = CollectionCandidate(
            image_id=11,
            path="photo.jpg",
            role_scores={
                "unknown_role": 0.99,
                "group": 0.50,
            },
        )

        matches = classify_candidate_roles(
            candidate,
            self.plan,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].role, "group")

    def test_empty_candidate(self):
        candidate = CollectionCandidate(
            image_id=12,
            path="photo.jpg",
        )

        self.assertEqual(
            classify_candidate_roles(candidate, self.plan),
            [],
        )

        self.assertIsNone(
            best_candidate_role(candidate, self.plan)
        )

    def test_tie_breaking_is_deterministic(self):
        candidate = CollectionCandidate(
            image_id=13,
            path="photo.jpg",
            role_scores={
                "group": 0.80,
                "activity": 0.80,
            },
            role_ranks={
                "group": 2,
                "activity": 1,
            },
        )

        matches = classify_candidate_roles(
            candidate,
            self.plan,
        )

        self.assertEqual(matches[0].role, "activity")


if __name__ == "__main__":
    unittest.main()
