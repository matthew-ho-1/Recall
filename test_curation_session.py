
import tempfile
import unittest
from pathlib import Path

from collection_candidates import CollectionCandidate
from collection_planner import create_collection_plan
from curation_session import (
    another_six,
    create_session,
    keep_photo,
    reject_photo,
    save_session,
    load_session,
)


class TestCurationSession(unittest.TestCase):

    def setUp(self):
        self.plan = create_collection_plan("Hinge")

        # Three possible photos for each of six roles.
        self.candidates = []

        for role_index, role in enumerate(self.plan.roles):
            for alternative in range(3):
                image_id = role_index * 10 + alternative + 1

                self.candidates.append(
                    CollectionCandidate(
                        image_id=image_id,
                        path=f"photo_{image_id}.jpg",
                        role_scores={
                            role.name: 0.9 - alternative * 0.1
                        },
                    )
                )

    def test_first_batch_contains_six_photos(self):
        session = create_session(self.plan)

        results = another_six(
            session, self.plan, self.candidates
        )

        self.assertEqual(len(results), 6)
        self.assertEqual(session.batch_number, 1)
        self.assertEqual(len(session.shown_ids), 6)

    def test_another_batch_has_no_repeats(self):
        session = create_session(self.plan)

        first = another_six(
            session, self.plan, self.candidates
        )
        second = another_six(
            session, self.plan, self.candidates
        )

        first_ids = {item.image_id for item in first}
        second_ids = {item.image_id for item in second}

        self.assertEqual(len(second), 6)
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_kept_photo_survives_next_batch(self):
        session = create_session(self.plan)

        first = another_six(
            session, self.plan, self.candidates
        )

        favorite = first[0]
        keep_photo(session, favorite.image_id)

        second = another_six(
            session, self.plan, self.candidates
        )

        self.assertIn(favorite, second)
        self.assertEqual(len(second), 6)

        other_first_ids = {
            item.image_id for item in first[1:]
        }
        other_second_ids = {
            item.image_id
            for item in second
            if item.image_id != favorite.image_id
        }

        self.assertTrue(
            other_first_ids.isdisjoint(other_second_ids)
        )

    def test_rejected_photo_does_not_return(self):
        session = create_session(self.plan)

        first = another_six(
            session, self.plan, self.candidates
        )

        rejected_id = first[0].image_id
        reject_photo(session, rejected_id)

        second = another_six(
            session, self.plan, self.candidates
        )

        self.assertNotIn(
            rejected_id,
            {item.image_id for item in second},
        )

    def test_rejecting_kept_photo_releases_role(self):
        session = create_session(self.plan)

        first = another_six(
            session, self.plan, self.candidates
        )

        favorite = first[0]
        keep_photo(session, favorite.image_id)
        reject_photo(session, favorite.image_id)

        self.assertNotIn(favorite.role, session.kept)

        second = another_six(
            session, self.plan, self.candidates
        )

        self.assertEqual(len(second), 6)
        self.assertNotIn(
            favorite.image_id,
            {item.image_id for item in second},
        )

    def test_exhaustion_returns_empty_collection(self):
        session = create_session(self.plan)

        for _ in range(3):
            self.assertEqual(
                len(another_six(
                    session, self.plan, self.candidates
                )),
                6,
            )

        fourth = another_six(
            session, self.plan, self.candidates
        )

        self.assertEqual(fourth, [])

    def test_session_save_and_load(self):
        session = create_session(self.plan)

        first = another_six(
            session, self.plan, self.candidates
        )
        keep_photo(session, first[0].image_id)
        reject_photo(session, first[1].image_id)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"

            save_session(session, path)
            restored = load_session(path)

        self.assertEqual(restored.goal, session.goal)
        self.assertEqual(restored.shown_ids, session.shown_ids)
        self.assertEqual(
            restored.rejected_ids,
            session.rejected_ids,
        )
        self.assertEqual(restored.kept, session.kept)
        self.assertEqual(restored.current, session.current)
        self.assertEqual(
            restored.batch_number,
            session.batch_number,
        )

    def test_cannot_keep_photo_not_in_current_batch(self):
        session = create_session(self.plan)

        with self.assertRaises(ValueError):
            keep_photo(session, 999)

    def test_cannot_reject_unseen_photo(self):
        session = create_session(self.plan)

        with self.assertRaises(ValueError):
            reject_photo(session, 999)

    def test_goal_mismatch_is_rejected(self):
        session = create_session(self.plan)
        session.goal = "different_goal"

        with self.assertRaises(ValueError):
            another_six(
                session, self.plan, self.candidates
            )


if __name__ == "__main__":
    unittest.main()
