from pathlib import Path

import numpy as np

from database import (
    connect,
    get_processed_images,
    get_searchable_faces,
)
from quality import (
    calculate_sharpness,
    calculate_face_sharpness,
)


DATABASE_PATH = Path(".recall/recall.db")
IDENTITY_PATH = Path(".recall/identities/me.npy")

SHARPNESS_LABELS = {
    "IMG_9583.jpg": "blurry",

    "105_4565.JPG": "sharp",
    "IMG_2745.HEIC": "sharp",
    "HMRR7967.JPG": "sharp",
    "797037703.069340.JPEG": "sharp",
    "IMG_0010.JPG": "sharp",

    "797018058.677324.JPG": "blurry",
    "797032995.853035.JPG": "blurry",
    "IMG_8728.HEIC": "blurry",
    "797017954.467796.JPEG": "blurry",
}


def calculate_pairwise_accuracy(
    results: list[tuple],
) -> tuple[int, int, float]:
    sharp_scores = []
    blurry_scores = []

    for (
        sharpness,
        image_id,
        original_path,
    ) in results:
        filename = Path(
            original_path
        ).name

        label = SHARPNESS_LABELS.get(
            filename
        )

        if label == "sharp":
            sharp_scores.append(
                sharpness
            )
        elif label == "blurry":
            blurry_scores.append(
                sharpness
            )

    correct = 0
    total = 0

    for sharp_score in sharp_scores:
        for blurry_score in blurry_scores:
            total += 1

            if sharp_score > blurry_score:
                correct += 1

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return (
        correct,
        total,
        accuracy,
    )


def load_normalized_embedding(
    path: Path,
) -> np.ndarray:
    embedding = np.load(
        path
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        raise ValueError(
            f"Embedding has zero norm: {path}"
        )

    return embedding / norm


def evaluate_whole_image_sharpness(
    connection,
) -> list[tuple]:
    images = get_processed_images(
        connection
    )

    results = []

    for (
        image_id,
        original_path,
        thumbnail_path,
    ) in images:
        filename = Path(
            original_path
        ).name

        if filename not in SHARPNESS_LABELS:
            continue

        thumbnail = Path(
            thumbnail_path
        )

        if not thumbnail.exists():
            continue

        sharpness = calculate_sharpness(
            thumbnail
        )

        results.append(
            (
                sharpness,
                image_id,
                original_path,
            )
        )

    return results


def evaluate_face_sharpness(
    connection,
) -> tuple[list[tuple], dict[int, float]]:
    identity_embedding = (
        load_normalized_embedding(
            IDENTITY_PATH
        )
    )

    faces = get_searchable_faces(
        connection
    )

    best_faces_by_image = {}

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
        filename = Path(
            image_path
        ).name

        if filename not in SHARPNESS_LABELS:
            continue

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
            best_faces_by_image[image_id] = {
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
    similarities = {}

    for (
        image_id,
        candidate,
    ) in best_faces_by_image.items():
        image_path = Path(
            candidate["image_path"]
        )

        sharpness = (
            calculate_face_sharpness(
                image_path,
                candidate["bbox"],
            )
        )

        results.append(
            (
                sharpness,
                image_id,
                str(image_path),
            )
        )

        similarities[image_id] = (
            candidate[
                "identity_similarity"
            ]
        )

    return (
        results,
        similarities,
    )


def print_labeled_results(
    title: str,
    results: list[tuple],
    similarities: dict[int, float] | None = None,
) -> None:
    print()
    print(title)
    print("-" * len(title))

    results = sorted(
        results,
        key=lambda result: result[0],
    )

    for (
        sharpness,
        image_id,
        original_path,
    ) in results:
        filename = Path(
            original_path
        ).name

        label = SHARPNESS_LABELS[
            filename
        ]

        identity_text = ""

        if similarities is not None:
            similarity = similarities.get(
                image_id
            )

            if similarity is not None:
                identity_text = (
                    f" identity={similarity:.4f}"
                )

        print(
            f"{label:6}  "
            f"{sharpness:10.2f}"
            f"{identity_text}  "
            f"[{image_id}]  "
            f"{filename}"
        )


def main() -> None:
    connection = connect(
        DATABASE_PATH
    )

    try:
        # ----------------------------------------
        # Baseline
        # ----------------------------------------

        whole_results = (
            evaluate_whole_image_sharpness(
                connection
            )
        )

        print_labeled_results(
            "Whole-image Laplacian",
            whole_results,
        )

        correct, total, accuracy = (
            calculate_pairwise_accuracy(
                whole_results
            )
        )

        print()
        print(
            "Whole-image pairwise accuracy: "
            f"{correct}/{total} "
            f"({accuracy:.1%})"
        )

        # ----------------------------------------
        # Candidate: standardized face region
        # ----------------------------------------

        (
            face_results,
            similarities,
        ) = evaluate_face_sharpness(
            connection
        )

        print_labeled_results(
            "Standardized face Laplacian",
            face_results,
            similarities,
        )

        correct, total, accuracy = (
            calculate_pairwise_accuracy(
                face_results
            )
        )

        coverage = (
            len(face_results)
            / len(SHARPNESS_LABELS)
        )

        print()
        print(
            "Face pairwise accuracy: "
            f"{correct}/{total} "
            f"({accuracy:.1%})"
        )

        print(
            "Face coverage: "
            f"{len(face_results)}/"
            f"{len(SHARPNESS_LABELS)} "
            f"({coverage:.1%})"
        )

        # ----------------------------------------
        # Missing labeled images
        # ----------------------------------------

        evaluated_filenames = {
            Path(
                original_path
            ).name
            for (
                sharpness,
                image_id,
                original_path,
            ) in face_results
        }

        missing = (
            set(SHARPNESS_LABELS)
            - evaluated_filenames
        )

        if missing:
            print()
            print(
                "No face-sharpness result"
            )
            print(
                "------------------------"
            )

            for filename in sorted(
                missing
            ):
                print(
                    f"{SHARPNESS_LABELS[filename]:6}  "
                    f"{filename}"
                )

    finally:
        connection.close()


if __name__ == "__main__":
    main()