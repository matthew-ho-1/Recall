from pathlib import Path
import sqlite3

import numpy as np

from database import get_searchable_faces
from faces import load_face_embedding
from identity import load_identity_embedding


def search_identity(
    connection: sqlite3.Connection,
    identity_path: Path,
    limit: int = 10,
) -> list[dict]:

    identity = load_identity_embedding(
        identity_path
    )

    faces = get_searchable_faces(
        connection
    )

    results_by_image = {}

    for (
        face_id,
        image_id,
        image_path,
        x1,
        y1,
        x2,
        y2,
        embedding_path,
    ) in faces:

        embedding_file = Path(
            embedding_path
        )

        if not embedding_file.exists():
            continue

        face_embedding = load_face_embedding(
            embedding_file
        )

        score = float(
            np.dot(
                identity,
                face_embedding,
            )
        )

        result = {
            "image_id": image_id,
            "path": image_path,
            "face_id": face_id,
            "score": score,
            "bbox": (
                x1,
                y1,
                x2,
                y2,
            ),
        }

        existing = results_by_image.get(
            image_id
        )

        if (
            existing is None
            or score > existing["score"]
        ):
            results_by_image[
                image_id
            ] = result

    results = list(
        results_by_image.values()
    )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:limit]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Search Recall for photos "
            "matching an identity."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results",
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
        results = search_identity(
            connection,
            identity_path,
            limit=args.limit,
        )

        print(
            "\n--- Identity search: me ---"
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"\n{rank}. "
                f"Score: "
                f"{result['score']:.4f}"
            )

            print(
                f"   {result['path']}"
            )

    finally:
        connection.close()