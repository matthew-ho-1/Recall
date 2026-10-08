
"""
Reusable search API for Recall.

Executes the existing identity-aware, quality-aware,
and diversity-aware search pipeline without printing
the final results.
"""

from pathlib import Path

from database import connect
from diversity_ranker import (
    select_diverse,
    select_mmr,
    select_without_near_duplicates,
)
from quality_search import (
    DATABASE_PATH,
    IDENTITY_PATH,
    load_candidate_embeddings,
    load_candidate_hashes,
    search_me_with_quality,
)


def search_photos(
    query: str,
    *,
    limit: int = 10,
    identity_limit: int = 100,
    pool_size: int = 30,
    rerank: bool = False,
    selection: str = "none",
    similarity_threshold: float = 0.90,
    max_hash_distance: int = 8,
    diversity_weight: float = 0.4,
    database_path: Path = DATABASE_PATH,
    identity_path: Path = IDENTITY_PATH,
) -> list[dict]:
    """Return ranked photo results without CLI formatting."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if limit < 1:
        raise ValueError("limit must be at least 1")

    if identity_limit < 1:
        raise ValueError("identity_limit must be at least 1")

    if pool_size < 1:
        raise ValueError("pool_size must be at least 1")

    if selection not in {
        "none",
        "diverse",
        "deduplicate",
        "mmr",
    }:
        raise ValueError(
            "selection must be none, diverse, deduplicate, or mmr"
        )

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between -1 and 1"
        )

    if not 0 <= max_hash_distance <= 63:
        raise ValueError(
            "max_hash_distance must be between 0 and 63"
        )

    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError(
            "diversity_weight must be between 0 and 1"
        )

    database_path = Path(database_path)
    identity_path = Path(identity_path)

    if not database_path.is_file():
        raise FileNotFoundError(database_path)

    if not identity_path.is_file():
        raise FileNotFoundError(identity_path)

    selection_enabled = rerank or selection != "none"

    candidate_limit = (
        max(pool_size, limit)
        if selection_enabled
        else limit
    )

    connection = connect(database_path)

    try:
        results = search_me_with_quality(
            connection,
            query,
            identity_path,
            identity_limit=identity_limit,
            candidate_limit=candidate_limit,
            rerank=rerank,
        )

        if selection in {"diverse", "mmr"}:
            embeddings = load_candidate_embeddings(
                connection,
                results,
            )
        else:
            embeddings = {}

        if selection == "deduplicate":
            hashes = load_candidate_hashes(results)
        else:
            hashes = {}

    finally:
        connection.close()

    if selection == "mmr":
        return select_mmr(
            results,
            embeddings,
            limit=limit,
            diversity_weight=diversity_weight,
        )

    if selection == "deduplicate":
        return select_without_near_duplicates(
            results,
            hashes,
            limit=limit,
            max_hash_distance=max_hash_distance,
        )

    if selection == "diverse":
        return select_diverse(
            results,
            embeddings,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

    return results[:limit]
