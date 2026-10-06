import sqlite3
from pathlib import Path

import torch
import time

from database import get_searchable_images
from embeddings import (
    embed_text,
    load_embedding,
    load_model,
)

def search(
    connection: sqlite3.Connection,
    query: str,
    limit: int = 5,
) -> list[dict]:

    model, _, tokenizer = load_model()

    query_embedding = embed_text(
        model,
        tokenizer,
        query,
    )

    images = get_searchable_images(
        connection
    )

    results = []

    for (
        image_id,
        image_path,
        thumbnail_path,
        embedding_path,
    ) in images:

        embedding_file = Path(
            embedding_path
        )

        if not embedding_file.exists():
            continue

        image_embedding = load_embedding(
            embedding_file
        )

        score = torch.dot(
            query_embedding,
            image_embedding,
        ).item()

        results.append(
            {
                "image_id": image_id,
                "path": image_path,
                "thumbnail_path": thumbnail_path,
                "score": score,
            }
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:limit]

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Search your Recall photo library."
        )
    )

    parser.add_argument(
        "query",
        help="Natural-language search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    args = parser.parse_args()

    database_path = (
        Path(".recall") / "recall.db"
    )

    if not database_path.exists():
        print(
            "Recall database not found. "
            "Run scanner.py first."
        )
        return

    connection = sqlite3.connect(
        database_path
    )

    try:
        start = time.perf_counter()
        results = search(
            connection,
            args.query,
            args.limit,
        )
        elapsed = time.perf_counter() - start
    finally:
        connection.close()

    print(
        f'\nResults for "{args.query}":\n'
    )

    if not results:
        print("No searchable images found.")
        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{index}. "
            f"{result['score']:.3f}  "
            f"{result['path']}"
        )
    print(f"\nSearch completed in {elapsed:.3f}s")


if __name__ == "__main__":
    main()