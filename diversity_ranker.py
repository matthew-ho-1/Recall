
import numpy as np

from perceptual_hash import hamming_distance


def select_diverse(
    results: list[dict],
    embeddings: dict[int, np.ndarray],
    limit: int = 10,
    similarity_threshold: float = 0.90,
) -> list[dict]:
    """
    Select varied photos from an already-ranked candidate list.

    Results should be ordered by the existing retrieval pipeline.
    Highly similar photos are deferred, then used as fallback if
    there are not enough distinct photos to fill the result set.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1 and 1")

    selected = []
    selected_vectors = []
    deferred = []

    for result in results:
        if len(selected) >= limit:
            break

        image_id = result["image_id"]
        embedding = embeddings.get(image_id)

        if embedding is None:
            deferred.append(result)
            continue

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)

        if not np.isfinite(norm) or norm == 0:
            deferred.append(result)
            continue

        vector = vector / norm

        too_similar = any(
            np.dot(vector, previous) >= similarity_threshold
            for previous in selected_vectors
        )

        if too_similar:
            deferred.append(result)
        else:
            selected.append(result)
            selected_vectors.append(vector)

    remaining_slots = limit - len(selected)

    if remaining_slots > 0:
        selected.extend(deferred[:remaining_slots])

    return selected



def select_without_near_duplicates(
    results: list[dict],
    hashes: dict[int, int],
    limit: int = 10,
    max_hash_distance: int = 8,
) -> list[dict]:
    """
    Prefer distinct photos from an already-ranked list.

    Defer candidates whose perceptual hashes are
    close to any already-selected photo.

    If there are too few distinct candidates,
    fill remaining slots in original ranking order.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")

    if not 0 <= max_hash_distance <= 63:
        raise ValueError(
            "max_hash_distance must be between 0 and 63"
        )

    selected = []
    selected_hashes = []
    deferred = []

    for result in results:
        if len(selected) >= limit:
            break

        image_id = result["image_id"]
        image_hash = hashes.get(image_id)

        if image_hash is None:
            deferred.append(result)
            continue

        near_duplicate = any(
            hamming_distance(
                image_hash,
                previous_hash,
            ) <= max_hash_distance
            for previous_hash in selected_hashes
        )

        if near_duplicate:
            deferred.append(result)
        else:
            selected.append(result)
            selected_hashes.append(image_hash)

    if len(selected) < limit:
        selected.extend(
            deferred[:limit - len(selected)]
        )

    return selected
