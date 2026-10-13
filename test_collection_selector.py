
import unittest

from collection_candidates import CollectionCandidate
from collection_planner import create_collection_plan
from collection_selector import select_collection


class TestCollectionSelector(unittest.TestCase):

    def setUp(self):
        self.plan = create_collection_plan("Hinge")

    def test_assigns_six_unique_images(self):
        candidates = []

        for index, role in enumerate(self.plan.roles):
            candidates.append(
                CollectionCandidate(
                    image_id=index + 1,
                    path=f"photo_{index + 1}.jpg",
                    role_scores={role.name: 0.9},
                )
            )

        selections = select_collection(self.plan, candidates)

        self.assertEqual(len(selections), 6)
        self.assertEqual(
            len({selection.image_id for selection in selections}),
            6,
        )

    def test_does_not_reuse_same_image(self):
        candidate = CollectionCandidate(
            image_id=42,
            path="photo.jpg",
            role_scores={
                role.name: 0.9
                for role in self.plan.roles
            },
        )

        selections = select_collection(self.plan, [candidate])

        self.assertEqual(len(selections), 1)

    def test_finds_better_global_assignment(self):
        candidates = [
            CollectionCandidate(
                image_id=1,
                path="photo1.jpg",
                role_scores={
                    "main_portrait": 0.95,
                    "group": 0.94,
                },
            ),
            CollectionCandidate(
                image_id=2,
                path="photo2.jpg",
                role_scores={
                    "main_portrait": 0.93,
                },
            ),
        ]

        selections = select_collection(self.plan, candidates)
        assignments = {
            selection.role: selection.image_id
            for selection in selections
        }

        self.assertEqual(assignments["main_portrait"], 2)
        self.assertEqual(assignments["group"], 1)

    def test_excludes_previous_images(self):
        candidates = [
            CollectionCandidate(
                image_id=1,
                path="photo1.jpg",
                role_scores={"group": 0.9},
            ),
            CollectionCandidate(
                image_id=2,
                path="photo2.jpg",
                role_scores={"group": 0.8},
            ),
        ]

        selections = select_collection(
            self.plan,
            candidates,
            excluded_ids={1},
        )

        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].image_id, 2)

    def test_empty_candidates(self):
        self.assertEqual(
            select_collection(self.plan, []),
            [],
        )

    def test_missing_roles_remain_unfilled(self):
        candidates = [
            CollectionCandidate(
                image_id=1,
                path="photo.jpg",
                role_scores={"activity": 0.8},
            ),
        ]

        selections = select_collection(self.plan, candidates)

        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].role, "activity")


if __name__ == "__main__":
    unittest.main()
