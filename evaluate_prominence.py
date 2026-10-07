from pathlib import Path

import numpy as np

from database import (
    connect,
    get_searchable_faces,
)
from quality import calculate_face_prominence


DATABASE_PATH = Path(".recall/recall.db")
IDENTITY_PATH = Path(".recall/identities/me.npy")


def load_normalized_embedding(
    path: Path,
) -> np.ndarray:
    embedding = np.load(path)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError(
            f"Embedding has zero norm: {path}"
        )

    return embedding / norm


def main() -> None:
    connection = connect(
        DATABASE_PATH
    )

    try:
        identity_embedding = (
            load_normalized_embedding(
                IDENTITY_PATH
            )
        )

        faces = get_searchable_faces(
            connection
        )

        best_faces_by_image = {}

        # Find the strongest identity match
        # in each image.
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
            face_embedding = (
                load_normalized_embedding(
                    Path(embedding_path)
                )
            )

            identity_similarity = float(
                np.dot(
                    identity_embedding,
                    face_embedding,
                )
            )

            current = best_faces_by_image.get(
                image_id
            )

            if (
                current is None
                or identity_similarity
                > current["identity_similarity"]
            ):
                best_faces_by_image[
                    image_id
                ] = {
                    "image_path": image_path,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "identity_similarity":
                        identity_similarity,
                }

        results = []

        for (
            image_id,
            candidate,
        ) in best_faces_by_image.items():
            image_path = Path(
                candidate["image_path"]
            )

            prominence = (
                calculate_face_prominence(
                    image_path,
                    candidate["bbox"],
                )
            )

            results.append(
                {
                    "image_id": image_id,
                    "path": image_path,
                    "identity_similarity":
                        candidate[
                            "identity_similarity"
                        ],
                    **prominence,
                }
            )

        results.sort(
            key=lambda result:
                result[
                    "face_area_fraction"
                ]
        )

        print()
        print("Lowest face prominence")
        print("----------------------")

        for result in results[:10]:
            print(
                f"area="
                f"{result['face_area_fraction']:6.2%}  "
                f"width="
                f"{result['face_width_fraction']:6.2%}  "
                f"height="
                f"{result['face_height_fraction']:6.2%}  "
                f"identity="
                f"{result['identity_similarity']:7.4f}  "
                f"[{result['image_id']}]  "
                f"{result['path'].name}"
            )

        print()
        print("Highest face prominence")
        print("-----------------------")

        for result in reversed(
            results[-10:]
        ):
            print(
                f"area="
                f"{result['face_area_fraction']:6.2%}  "
                f"width="
                f"{result['face_width_fraction']:6.2%}  "
                f"height="
                f"{result['face_height_fraction']:6.2%}  "
                f"identity="
                f"{result['identity_similarity']:7.4f}  "
                f"[{result['image_id']}]  "
                f"{result['path'].name}"
            )

    finally:
        connection.close()


if __name__ == "__main__":
    main()