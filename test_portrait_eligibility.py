
import unittest

from collection_candidates import CollectionCandidate
from portrait_eligibility import (
    PortraitEvidence,
    is_eligible_main_portrait,
    filter_main_portrait_candidates,
)

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

class TestPortraitEligibility(unittest.TestCase):

    def test_single_target_face_is_eligible(self):
        evidence = PortraitEvidence(
            face_count=1,
            target_identity_present=True,
        )

        self.assertTrue(
            is_eligible_main_portrait(evidence)
        )

    def test_group_photo_is_ineligible(self):
        evidence = PortraitEvidence(
            face_count=3,
            target_identity_present=True,
        )

        self.assertFalse(
            is_eligible_main_portrait(evidence)
        )

    def test_single_non_target_face_is_ineligible(self):
        evidence = PortraitEvidence(
            face_count=1,
            target_identity_present=False,
        )

        self.assertFalse(
            is_eligible_main_portrait(evidence)
        )

    def test_missing_face_is_ineligible(self):
        evidence = PortraitEvidence(
            face_count=0,
            target_identity_present=False,
        )

        self.assertFalse(
            is_eligible_main_portrait(evidence)
        )

    def test_filter_preserves_other_roles(self):
        candidate = CollectionCandidate(
            image_id=18,
            path="HMRR7967.JPG",
            role_scores={
                "main_portrait": 0.202,
                "group": 0.285,
            },
            role_ranks={
                "main_portrait": 1,
                "group": 2,
            },
        )

        evidence = {
            18: PortraitEvidence(
                face_count=3,
                target_identity_present=True,
            )
        }

        filter_main_portrait_candidates(
            [candidate],
            evidence,
        )

        self.assertNotIn(
            "main_portrait",
            candidate.role_scores,
        )
        self.assertIn(
            "group",
            candidate.role_scores,
        )

    def test_missing_evidence_is_not_eligible(self):
        candidate = CollectionCandidate(
            image_id=99,
            path="unknown.jpg",
            role_scores={"main_portrait": 0.8},
        )

        filter_main_portrait_candidates(
            [candidate],
            {},
        )

        self.assertNotIn(
            "main_portrait",
            candidate.role_scores,
        )

        
    def test_database_group_face_count(self):
        connection = sqlite3.connect(":memory:")

        try:
            connection.execute(
                """
                CREATE TABLE faces (
                    id INTEGER PRIMARY KEY,
                    image_id INTEGER NOT NULL,
                    embedding_path TEXT NOT NULL
                )
                """
            )

            connection.executemany(
                """
                INSERT INTO faces (image_id, embedding_path)
                VALUES (?, ?)
                """,
                [
                    (18, "face_1.npy"),
                    (18, "face_2.npy"),
                    (18, "face_3.npy"),
                ],
            )

            with patch(
                "portrait_eligibility.load_identity_embedding"
            ) as mock_identity:
                mock_identity.return_value = None

                from portrait_eligibility import (
                    load_portrait_evidence,
                )

                evidence = load_portrait_evidence(
                    connection,
                    Path("identity.npy"),
                    {18},
                )

            self.assertEqual(evidence[18].face_count, 3)
            self.assertFalse(
                evidence[18].target_identity_present
            )
        finally:
            connection.close()



if __name__ == "__main__":
    unittest.main()
