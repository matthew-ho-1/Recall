
import unittest

import numpy as np

from diversity_ranker import (
    select_diverse,
    select_without_near_duplicates,
    select_mmr,
)


class TestNearDuplicateSelection(unittest.TestCase):

    def setUp(self):
        self.results = [
            {"image_id": 1, "path": "portrait.jpg"},
            {"image_id": 2, "path": "portrait_copy.jpg"},
            {"image_id": 3, "path": "hiking.jpg"},
            {"image_id": 4, "path": "friends.jpg"},
        ]

        self.hashes = {
            1: 0b0000000000,
            2: 0b0000000011,
            3: 0b1111111111,
            4: 0b1010101010,
        }

    def test_defers_near_duplicates(self):
        selected = select_without_near_duplicates(
            self.results,
            self.hashes,
            limit=3,
            max_hash_distance=2,
        )

        self.assertEqual(
            [item["image_id"] for item in selected],
            [1, 3, 4],
        )

    def test_preserves_distinct_results(self):
        selected = select_without_near_duplicates(
            self.results,
            self.hashes,
            limit=4,
            max_hash_distance=0,
        )

        self.assertEqual(
            [item["image_id"] for item in selected],
            [1, 2, 3, 4],
        )

    def test_fills_remaining_slots(self):
        selected = select_without_near_duplicates(
            self.results,
            self.hashes,
            limit=4,
            max_hash_distance=2,
        )

        self.assertEqual(
            [item["image_id"] for item in selected],
            [1, 3, 4, 2],
        )

    def test_missing_hash_does_not_crash(self):
        hashes = {
            1: self.hashes[1],
            3: self.hashes[3],
        }

        selected = select_without_near_duplicates(
            self.results,
            hashes,
            limit=3,
            max_hash_distance=2,
        )

        self.assertEqual(
            [item["image_id"] for item in selected],
            [1, 3, 2],
        )

    def test_zero_limit(self):
        selected = select_without_near_duplicates(
            self.results,
            self.hashes,
            limit=0,
        )

        self.assertEqual(selected, [])

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            select_without_near_duplicates(
                self.results,
                self.hashes,
                max_hash_distance=64,
            )


class TestOpenCLIPDiversity(unittest.TestCase):

    def test_defers_similar_embeddings(self):
        results = [
            {"image_id": 1},
            {"image_id": 2},
            {"image_id": 3},
        ]

        embeddings = {
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.99, 0.01, 0.0]),
            3: np.array([0.0, 1.0, 0.0]),
        }

        selected = select_diverse(
            results,
            embeddings,
            limit=2,
            similarity_threshold=0.90,
        )

        self.assertEqual(
            [item["image_id"] for item in selected],
            [1, 3],
        )


class TestMMRSelection(unittest.TestCase):

    def setUp(self):
        self.results = [
            {"image_id": 1, "quality_tier": 0},
            {"image_id": 2, "quality_tier": 0},
            {"image_id": 3, "quality_tier": 0},
            {"image_id": 4, "quality_tier": 0},
        ]

        self.embeddings = {
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.99, 0.01, 0.0]),
            3: np.array([0.0, 1.0, 0.0]),
            4: np.array([0.0, 0.0, 1.0]),
        }

    def test_zero_weight_preserves_ranking(self):
        results = [
            {"image_id": 1, "quality_tier": 2},
            {"image_id": 2, "quality_tier": 0},
            {"image_id": 3, "quality_tier": 0},
        ]

        selected = select_mmr(
            results,
            self.embeddings,
            limit=3,
            diversity_weight=0.0,
        )

        self.assertEqual(
            [item["image_id"] for item in selected],
            [1, 2, 3],
        )

    def test_diversity_promotes_distinct_photos(self):
        selected = select_mmr(
            self.results,
            self.embeddings,
            limit=3,
            diversity_weight=0.6,
        )

        self.assertEqual(
            [item["image_id"] for item in selected],
            [1, 3, 4],
        )

    def test_zero_limit(self):
        selected = select_mmr(
            self.results,
            self.embeddings,
            limit=0,
        )

        self.assertEqual(selected, [])

    def test_invalid_weight(self):
        with self.assertRaises(ValueError):
            select_mmr(
                self.results,
                self.embeddings,
                diversity_weight=1.5,
            )

    def test_missing_embedding(self):
        selected = select_mmr(
            self.results,
            {1: self.embeddings[1]},
            limit=3,
            diversity_weight=0.4,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            len({item["image_id"] for item in selected}),
            3,
        )



    def test_quality_penalty(self):
        results = [
            {"image_id": 1, "quality_tier": 0},
            {"image_id": 2, "quality_tier": 2},
            {"image_id": 3, "quality_tier": 0},
            {"image_id": 4, "quality_tier": 0},
            {"image_id": 5, "quality_tier": 0},
            {"image_id": 6, "quality_tier": 0},
            {"image_id": 7, "quality_tier": 0},
            {"image_id": 8, "quality_tier": 0},
            {"image_id": 9, "quality_tier": 0},
            {"image_id": 10, "quality_tier": 0},
        ]

        # Image 1 is visually distinct.
        # Images 2-10 have identical embeddings.
        embeddings = {
            image_id: np.array([0.0, 1.0, 0.0])
            for image_id in range(2, 11)
        }

        embeddings[1] = np.array([1.0, 0.0, 0.0])

        # Control group: all candidates have tier 0.
        no_penalty_results = [
            {**result, "quality_tier": 0}
            for result in results
        ]

        without_penalty = select_mmr(
            no_penalty_results,
            embeddings,
            limit=3,
            diversity_weight=0.4,
        )

        # Experimental group: image 2 has tier 2.
        with_penalty = select_mmr(
            results,
            embeddings,
            limit=3,
            diversity_weight=0.4,
        )

        without_ids = [
            item["image_id"]
            for item in without_penalty
        ]

        with_ids = [
            item["image_id"]
            for item in with_penalty
        ]

        self.assertEqual(
            without_ids,
            [1, 2, 3],
        )

        self.assertEqual(
            with_ids,
            [1, 3, 2],
        )

if __name__ == "__main__":
    unittest.main()