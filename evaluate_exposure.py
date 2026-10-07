from pathlib import Path

import numpy as np

from database import (
    connect,
    get_searchable_faces,
)
from quality import (
    calculate_exposure,
    calculate_region_exposure,
)


DATABASE_PATH = Path(".recall/recall.db")
IDENTITY_PATH = Path(".recall/identities/me.npy")


EXPOSURE_LABELS = {
    # Properly exposed
    "IMG_4695.HEIC": "good",
    "IMG_6255.JPG": "good",
    "IMG_7892.JPG": "good",

    # Overexposed
    "797018058.677324.JPG": "overexposed",
    "797032940.345699.JPG": "overexposed",
    "797032995.853035.JPG": "overexposed",

    # Underexposed
    "IMG_2792.HEIC": "underexposed",
    "under2.jpg": "underexposed",
    "under3.jpg": "underexposed",
}


def load_normalized_embedding(
    path: Path,
) -> np.ndarray:
    embedding = np.load(path)

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        raise ValueError(
            f"Embedding has zero norm: {path}"
        )

    return embedding / norm


def calculate_pairwise_accuracy(
    positive_scores: list[float],
    negative_scores: list[float],
) -> tuple[int, int]:
    """
    Count how often a positive example scores higher
    than a negative example.
    """
    correct = 0
    total = 0

    for positive in positive_scores:
        for negative in negative_scores:
            total += 1

            if positive > negative:
                correct += 1

    return correct, total


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

        # ----------------------------------------
        # Find the strongest identity match
        # in each labeled image.
        #
        # We intentionally do not apply an identity
        # threshold here. Very poor lighting may
        # reduce identity similarity, and we want
        # those images represented in the exposure
        # experiment.
        # ----------------------------------------

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

            if filename not in EXPOSURE_LABELS:
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

            current = (
                best_faces_by_image.get(
                    image_id
                )
            )

            if (
                current is None
                or identity_similarity
                > current["identity_similarity"]
            ):
                best_faces_by_image[
                    image_id
                ] = {
                    "image_path":
                        image_path,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "identity_similarity":
                        identity_similarity,
                }

        # ----------------------------------------
        # Collect benchmark measurements.
        # ----------------------------------------

        good_whole_means = []
        under_whole_means = []

        good_face_means = []
        over_face_means = []

        evaluated = set()

        # ----------------------------------------
        # Compare whole-image exposure with
        # face-region exposure.
        # ----------------------------------------

        print()
        print("Exposure comparison")
        print("-------------------")

        for (
            image_id,
            candidate,
        ) in best_faces_by_image.items():
            image_path = Path(
                candidate["image_path"]
            )

            filename = image_path.name

            label = EXPOSURE_LABELS[
                filename
            ]

            whole = calculate_exposure(
                image_path
            )

            face = calculate_region_exposure(
                image_path,
                candidate["bbox"],
            )

            evaluated.add(
                filename
            )

            # ------------------------------------
            # Collect measurements for the two
            # candidate exposure signals.
            #
            # Underexposure:
            # good whole-image mean should be
            # HIGHER than underexposed mean.
            #
            # Overexposure:
            # overexposed face mean should be
            # HIGHER than good face mean.
            # ------------------------------------

            if label == "good":
                good_whole_means.append(
                    whole["mean_brightness"]
                )

                good_face_means.append(
                    face["mean_brightness"]
                )

            elif label == "underexposed":
                under_whole_means.append(
                    whole["mean_brightness"]
                )

            elif label == "overexposed":
                over_face_means.append(
                    face["mean_brightness"]
                )

            # ------------------------------------
            # Print diagnostic measurements.
            # ------------------------------------

            print()
            print(
                f"{filename} [{label}]"
            )

            print(
                f"  identity: "
                f"{candidate['identity_similarity']:.4f}"
            )

            print(
                "  whole: "
                f"mean={whole['mean_brightness']:.2f}  "
                f"dark={whole['dark_fraction']:.1%}  "
                f"bright={whole['bright_fraction']:.1%}"
            )

            print(
                "  face:  "
                f"mean={face['mean_brightness']:.2f}  "
                f"dark={face['dark_fraction']:.1%}  "
                f"bright={face['bright_fraction']:.1%}"
            )

        # ----------------------------------------
        # Report labeled images for which no
        # searchable face was available.
        # ----------------------------------------

        missing = (
            set(EXPOSURE_LABELS)
            - evaluated
        )

        if missing:
            print()
            print("No face result")
            print("--------------")

            for filename in sorted(
                missing
            ):
                print(filename)

        # ----------------------------------------
        # Pairwise benchmark.
        #
        # This measures separation on the current
        # labeled sample. It is NOT a claim of
        # general exposure-classification accuracy.
        # ----------------------------------------

        print()
        print("Exposure benchmark")
        print("------------------")

        if (
            good_whole_means
            and under_whole_means
        ):
            correct, total = (
                calculate_pairwise_accuracy(
                    good_whole_means,
                    under_whole_means,
                )
            )

            print(
                f"Underexposure / whole mean: "
                f"{correct}/{total} "
                f"({correct / total:.1%})"
            )
        else:
            print(
                "Underexposure / whole mean: "
                "insufficient data"
            )

        if (
            over_face_means
            and good_face_means
        ):
            correct, total = (
                calculate_pairwise_accuracy(
                    over_face_means,
                    good_face_means,
                )
            )

            print(
                f"Overexposure / face mean: "
                f"{correct}/{total} "
                f"({correct / total:.1%})"
            )
        else:
            print(
                "Overexposure / face mean: "
                "insufficient data"
            )

    finally:
        connection.close()


if __name__ == "__main__":
    main()