from pathlib import Path
import sqlite3

import torch

from database import get_searchable_images
from embeddings import (
    embed_text,
    load_embedding,
    load_model,
)
from identity_search import search_identity
import time


def search_me_semantic(
    connection: sqlite3.Connection,
    query: str,
    identity_path: Path,
    identity_limit: int = 100,
    limit: int = 10,
) -> list[dict]:

    # --------------------------------------------------
    # 1. Find candidate photos containing "me"
    # --------------------------------------------------

    identity_results = search_identity(
        connection,
        identity_path,
        limit=identity_limit,
    )

    # --------------------------------------------------
    # 2. Load CLIP and embed the semantic query
    # --------------------------------------------------

    model, _, tokenizer = load_model()

    query_embedding = embed_text(
        model,
        tokenizer,
        query,
    )

    # --------------------------------------------------
    # 3. Build image_id -> CLIP embedding lookup
    # --------------------------------------------------

    searchable_images = get_searchable_images(
        connection
    )

    embedding_paths = {
        image_id: embedding_path
        for (
            image_id,
            image_path,
            thumbnail_path,
            embedding_path,
        ) in searchable_images
    }

    # --------------------------------------------------
    # 4. Semantic-score only the identity candidates
    # --------------------------------------------------

    results = []

    for identity_result in identity_results:
        image_id = identity_result[
            "image_id"
        ]

        embedding_path = embedding_paths.get(
            image_id
        )

        if embedding_path is None:
            continue

        embedding_file = Path(
            embedding_path
        )

        if not embedding_file.exists():
            continue

        image_embedding = load_embedding(
            embedding_file
        )

        semantic_score = torch.dot(
            query_embedding,
            image_embedding,
        ).item()

        results.append(
            {
                "image_id": image_id,
                "path": identity_result["path"],
                "identity_score": (
                    identity_result["score"]
                ),
                "semantic_score": semantic_score,
            }
        )

    # --------------------------------------------------
    # 5. Rank by semantic relevance
    # --------------------------------------------------

    results.sort(
        key=lambda result: (
            result["semantic_score"]
        ),
        reverse=True,
    )

    return results[:limit]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Search Recall for photos of me "
            "matching a semantic query."
        )
    )

    parser.add_argument(
        "query",
        type=str,
        help="Semantic search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results",
    )

    parser.add_argument(
        "--identity-limit",
        type=int,
        default=100,
        help=(
            "Number of identity candidates "
            "to consider"
        ),
    )

    args = parser.parse_args()

    database_path = Path(
        ".recall/recall.db"
    )

    identity_path = Path(
        ".recall/identities/me.npy"
    )

    if not database_path.exists():
        raise FileNotFoundError(
            f"Database not found: "
            f"{database_path}"
        )

    if not identity_path.exists():
        raise FileNotFoundError(
            f"Identity not found: "
            f"{identity_path}"
        )

    connection = sqlite3.connect(
        database_path
    )

    try:
        start_time = time.perf_counter()
        results = search_me_semantic(
            connection,
            args.query,
            identity_path,
            identity_limit=args.identity_limit,
            limit=args.limit,
        )
        elapsed = time.perf_counter() - start_time
        print(
            f'\n--- Photos of me: '
            f'"{args.query}" ---'
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"\n{rank}. "
                f"Semantic: "
                f"{result['semantic_score']:.4f} "
                f"| Identity: "
                f"{result['identity_score']:.4f}"
            )

            print(
                f"   {result['path']}"
            )

            print(
                f"\nSearch time: "
                f"{elapsed:.3f} seconds"
            )

    finally:
        connection.close()