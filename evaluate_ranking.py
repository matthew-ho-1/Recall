
from pathlib import Path

import numpy as np

from database import (
    connect,
    get_searchable_faces,
)
from quality import (
    calculate_exposure,
    calculate_face_prominence,
    calculate_face_sharpness,
    calculate_resolution,
    calculate_region_exposure,
)
from quality_ranker import (
    evaluate_quality,
    quality_tier,
)


DATABASE_PATH = Path(".recall/recall.db")
IDENTITY_PATH = Path(".recall/identities/me.npy")


QUALITY_LABELS = {
    # Strong
    "IMG_4695.HEIC": "strong",
    "HMRR7967.JPG": "strong",
    "IMG_2745.HEIC": "strong",
    "IMG_6255.JPG": "strong",
    "IMG_6651.HEIC": "strong",
    "IMG_6971.HEIC": "strong",

    # Acceptable
    "IMG_0010.JPG": "acceptable",
    "IMG_2792.HEIC": "acceptable",

    # Weak
    "IMG_5861.JPG": "weak",
    "IMG_9583.jpg": "weak",
    "797032940.345699.JPG": "weak",
    "797018058.677324.JPG": "weak",
}


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
    connection = connect(DATABASE_PATH)

    try:
        identity_embedding = load_normalized_embedding(
            IDENTITY_PATH
        )

        faces = get_searchable_faces(connection)
        best_faces_by_image = {}

        # ----------------------------------------
        # Select the strongest identity match
        # for each labeled image.
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
            filename = Path(image_path).name

            if filename not in QUALITY_LABELS:
                continue

            face_embedding = load_normalized_embedding(
                Path(embedding_path)
            )

            identity_similarity = float(
                np.dot(
                    identity_embedding,
                    face_embedding,
                )
            )

            current = best_faces_by_image.get(image_id)

            if (
                current is None
                or identity_similarity
                > current["identity_similarity"]
            ):
                best_faces_by_image[image_id] = {
                    "image_path": image_path,
                    "bbox": (x1, y1, x2, y2),
                    "identity_similarity":
                        identity_similarity,
                }

        results = []

        # ----------------------------------------
        # Calculate v0.6 quality measurements.
        # ----------------------------------------

        for image_id, candidate in (
            best_faces_by_image.items()
        ):
            image_path = Path(
                candidate["image_path"]
            )

            filename = image_path.name
            bbox = candidate["bbox"]

            whole_exposure = calculate_exposure(
                image_path
            )

            face_exposure = calculate_region_exposure(
                image_path,
                bbox,
            )

            resolution = calculate_resolution(
                image_path
            )

            prominence = calculate_face_prominence(
                image_path,
                bbox,
            )

            face_sharpness = calculate_face_sharpness(
                image_path,
                bbox,
            )

            quality = evaluate_quality(
                face_sharpness=face_sharpness,
                whole_mean_brightness=(
                    whole_exposure["mean_brightness"]
                ),
                face_mean_brightness=(
                    face_exposure["mean_brightness"]
                ),
                megapixels=(
                    resolution["megapixels"]
                ),
                face_prominence=(
                    prominence["face_area_fraction"]
                ),
            )

            tier = quality_tier(quality)

            results.append(
                {
                    "image_id": image_id,
                    "filename": filename,
                    "label": QUALITY_LABELS[filename],
                    "identity": candidate[
                        "identity_similarity"
                    ],
                    "sharpness": face_sharpness,
                    "whole_mean": whole_exposure[
                        "mean_brightness"
                    ],
                    "face_mean": face_exposure[
                        "mean_brightness"
                    ],
                    "megapixels": resolution[
                        "megapixels"
                    ],
                    "prominence": prominence[
                        "face_area_fraction"
                    ],
                    "blur_evidence":
                        quality.blur_evidence,
                    "underexposure_evidence":
                        quality.underexposure_evidence,
                    "overexposure_evidence":
                        quality.overexposure_evidence,
                    "defects":
                        quality.technical_defect_count,
                    "tier": tier,
                }
            )

        label_order = [
            "strong",
            "acceptable",
            "weak",
        ]

        # ----------------------------------------
        # Print detailed results by human label.
        # ----------------------------------------

        for label in label_order:
            print()
            print(label.upper())
            print("=" * len(label))

            label_results = [
                result
                for result in results
                if result["label"] == label
            ]

            for result in label_results:
                print()
                print(result["filename"])

                print(
                    f"  identity:    "
                    f"{result['identity']:.4f}"
                )

                print(
                    f"  sharpness:   "
                    f"{result['sharpness']:.2f}"
                )

                print(
                    f"  whole mean:  "
                    f"{result['whole_mean']:.2f}"
                )

                print(
                    f"  face mean:   "
                    f"{result['face_mean']:.2f}"
                )

                print(
                    f"  resolution:  "
                    f"{result['megapixels']:.2f} MP"
                )

                print(
                    f"  prominence:  "
                    f"{result['prominence']:.2%}"
                )

                print(
                    f"  defects:     "
                    f"{result['defects']}"
                )

                print(
                    f"  blur:        "
                    f"{result['blur_evidence']}"
                )

                print(
                    f"  underexp:    "
                    f"{result['underexposure_evidence']}"
                )

                print(
                    f"  overexp:     "
                    f"{result['overexposure_evidence']}"
                )

                print(
                    f"  tier:        "
                    f"{result['tier']}"
                )

        # ----------------------------------------
        # Report images without a searchable face.
        # ----------------------------------------

        evaluated = {
            result["filename"]
            for result in results
        }

        missing = set(QUALITY_LABELS) - evaluated

        if missing:
            print()
            print("NO FACE RESULT")
            print("==============")

            for filename in sorted(missing):
                print(filename)

        # ----------------------------------------
        # Defect summary by human quality label.
        # ----------------------------------------

        print()
        print("DEFECT SUMMARY")
        print("==============")

        for label in label_order:
            label_results = [
                result
                for result in results
                if result["label"] == label
            ]

            clean_count = sum(
                result["defects"] == 0
                for result in label_results
            )

            flagged_count = sum(
                result["defects"] > 0
                for result in label_results
            )

            print(
                f"{label:<10} "
                f"clean={clean_count}  "
                f"flagged={flagged_count}"
            )

        # ----------------------------------------
        # Quality-aware ranking.
        #
        # Lower tier means fewer detected
        # technical defects.
        # ----------------------------------------

        print()
        print("QUALITY-AWARE RANKING")
        print("=====================")

        ranked_results = sorted(
            results,
            key=lambda result: result["tier"],
        )

        for index, result in enumerate(
            ranked_results,
            start=1,
        ):
            print(
                f"{index:2d}. "
                f"tier={result['tier']}  "
                f"defects={result['defects']}  "
                f"human={result['label']:<10}  "
                f"{result['filename']}"
            )

        # ----------------------------------------
        # Tier distribution.
        # ----------------------------------------

        print()
        print("TIER DISTRIBUTION")
        print("=================")

        for tier in (0, 1, 2):
            count = sum(
                result["tier"] == tier
                for result in results
            )

            print(f"Tier {tier}: {count} photos")

        print()
        print(f"Evaluated: {len(results)} photos")
        print(f"Missing face result: {len(missing)} photos")

        # ----------------------------------------
        # v0.6 regression checks.
        #
        # Verify known labeled examples.
        # These are in-sample regression checks,
        # not a test of generalization.
        # ----------------------------------------

        by_filename = {
            result["filename"]: result
            for result in results
        }

        strong_photos = [
            filename
            for filename, label in QUALITY_LABELS.items()
            if label == "strong"
        ]

        assert all(
            filename in by_filename
            and by_filename[filename]["tier"] == 0
            for filename in strong_photos
        ), "A known strong photo was penalized"

        assert (
            by_filename[
                "797018058.677324.JPG"
            ]["tier"] == 2
        ), (
            "Known blurry and overexposed "
            "photo was not flagged correctly"
        )

        assert (
            by_filename[
                "797032940.345699.JPG"
            ]["tier"] == 1
        ), (
            "Known overexposed photo "
            "was not flagged correctly"
        )

        assert (
            "IMG_9583.jpg" in missing
        ), (
            "Expected missing-face "
            "example changed"
        )

        assert len(results) == 11, (
            "Unexpected number of evaluated photos"
        )

        print()
        print("v0.6 REGRESSION CHECKS PASSED")

    finally:
        connection.close()

if __name__ == "__main__":
    main()
