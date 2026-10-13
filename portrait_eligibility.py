
"""
Portrait eligibility rules for Recall v0.10.

Uses face detections to decide whether a photo can fill
the main_portrait role.
"""

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np

from faces import load_face_embedding
from identity import load_identity_embedding


@dataclass(frozen=True)
class PortraitEvidence:
    face_count: int
    target_identity_present: bool


def is_eligible_main_portrait(
    evidence: PortraitEvidence,
) -> bool:
    """
    A main portrait requires exactly one detected face,
    belonging to the target identity.
    """

    return (
        evidence.face_count == 1
        and evidence.target_identity_present
    )


def filter_main_portrait_candidates(
    candidates,
    evidence_by_image: dict[int, PortraitEvidence],
):
    """
    Remove main_portrait eligibility from unsuitable photos.

    Preserve each candidate's other role scores so that
    a group photo can still be selected for another role.
    """

    for candidate in candidates:
        evidence = evidence_by_image.get(candidate.image_id)

        if (
            evidence is None
            or not is_eligible_main_portrait(evidence)
        ):
            candidate.role_scores.pop("main_portrait", None)
            candidate.role_ranks.pop("main_portrait", None)
            candidate.search_results.pop("main_portrait", None)

    return candidates


def load_portrait_evidence(
    connection: sqlite3.Connection,
    identity_path: Path,
    image_ids: set[int],
    *,
    identity_threshold: float = 0.35,
) -> dict[int, PortraitEvidence]:
    """
    Load portrait eligibility evidence from indexed faces.

    Check every detected face for the target identity.

    A photo may contain multiple faces and still contain
    the target person. The main_portrait eligibility rule
    separately requires exactly one detected face.
    """

    if not -1.0 <= identity_threshold <= 1.0:
        raise ValueError(
            "identity_threshold must be between -1 and 1."
        )

    if not image_ids:
        return {}

    identity_embedding = load_identity_embedding(
        Path(identity_path)
    )

    placeholders = ",".join("?" for _ in image_ids)

    rows = connection.execute(
        f"""
        SELECT
            image_id,
            embedding_path
        FROM faces
        WHERE image_id IN ({placeholders})
        ORDER BY image_id, id
        """,
        tuple(sorted(image_ids)),
    ).fetchall()

    faces_by_image = {}

    for image_id, embedding_path in rows:
        faces_by_image.setdefault(
            image_id,
            [],
        ).append(embedding_path)

    evidence_by_image = {}

    for image_id in image_ids:
        face_paths = faces_by_image.get(
            image_id,
            [],
        )

        face_count = len(face_paths)
        target_present = False

        # Check all detected faces, including group photos.
        for stored_path in face_paths:
            embedding_path = Path(stored_path)

            if not embedding_path.is_file():
                continue

            face_embedding = load_face_embedding(
                embedding_path
            )

            similarity = float(
                np.dot(
                    identity_embedding,
                    face_embedding,
                )
            )

            if similarity >= identity_threshold:
                target_present = True
                break

        evidence_by_image[image_id] = PortraitEvidence(
            face_count=face_count,
            target_identity_present=target_present,
        )

    return evidence_by_image
