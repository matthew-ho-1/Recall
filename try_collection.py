
"""
Temporary smoke test for Recall v0.10 collection generation.

Searches the real indexed library and prints a six-photo
collection without modifying photos or session state.
"""

from collections import Counter

from collection_planner import create_collection_plan
from collection_candidates import retrieve_collection_candidates
from collection_selector import select_collection


from database import connect
from search_api import DATABASE_PATH, IDENTITY_PATH
from portrait_eligibility import (
    load_portrait_evidence,
    filter_main_portrait_candidates,
)

from pathlib import Path

import numpy as np

from collection_candidates import CollectionCandidate
from portrait_eligibility import load_identity_embedding

# User-confirmed photos that do not contain the target person.
# Temporary until Recall has persistent identity feedback.
# Photos confirmed by the user to NOT contain the target person.
# Temporary feedback list until Recall supports persistent labels.
REJECTED_IDENTITY_IMAGE_IDS = {
    # First collection
    3401,  # IMG_E4697.HEIC
    2186,  # IMG_E6294.JPG
    6128,  # IMG_1127.JPG

    # Second collection
    3403,  # IMG_E4696.HEIC
    3404,  # IMG_4696.HEIC
    6129,  # IMG_1128.JPG
    2180,  # IMG_E6297.JPG
}

def add_eligible_portrait_candidates(
    connection,
    candidates,
    identity_path,
    threshold=0.35,
):
    """
    Add solo-face identity matches that semantic retrieval missed.

    These candidates are ranked by their existing CLIP image
    embeddings against a portrait text query.
    """
    import torch
    from embeddings import load_model

    identity = load_identity_embedding(Path(identity_path))

    rows = connection.execute(
        """
        SELECT
            images.id,
            images.path,
            images.embedding_path,
            faces.embedding_path
        FROM images
        JOIN faces ON faces.image_id = images.id
        WHERE images.embedding_path IS NOT NULL
          AND images.processing_error IS NULL
          AND images.id IN (
              SELECT image_id
              FROM faces
              GROUP BY image_id
              HAVING COUNT(*) = 1
          )
        """
    ).fetchall()

    eligible = []

    for image_id, path, clip_path, face_path in rows:
        face_file = Path(face_path)
        clip_file = Path(clip_path)

        if not face_file.is_file() or not clip_file.is_file():
            continue

        face_embedding = np.load(face_file).reshape(-1)
        similarity = float(np.dot(identity, face_embedding))

        if similarity >= threshold:
            eligible.append((image_id, path, clip_file))

    print(f"Eligible solo portraits found: {len(eligible)}")

    if not eligible:
        return candidates

    # Use the same OpenCLIP model as Recall's indexing pipeline.
    model, _, tokenizer = load_model()

    query = (
        "a flattering, clear, well-lit solo portrait photo "
        "of one person smiling at the camera"
    )

    with torch.no_grad():
        tokens = tokenizer([query])
        text_embedding = model.encode_text(tokens)
        text_embedding = torch.nn.functional.normalize(
            text_embedding,
            dim=-1,
        ).cpu().numpy().reshape(-1)

    existing = {
        candidate.image_id: candidate
        for candidate in candidates
    }

    for image_id, path, clip_file in eligible:
        image_embedding = np.load(clip_file).reshape(-1)
        image_embedding /= np.linalg.norm(image_embedding)

        score = float(
            np.dot(text_embedding, image_embedding)
        )

        if image_id not in existing:
            candidate = CollectionCandidate(
                image_id=image_id,
                path=str(path),
            )
            candidates.append(candidate)
            existing[image_id] = candidate

        existing[image_id].role_scores["main_portrait"] = score

    return candidates

def main():
    plan = create_collection_plan(
        "Find the best 6 photos for my Hinge profile"
    )

    print("Searching indexed photo library...")
    print(f"Collection roles: {plan.count}")

    candidates = retrieve_collection_candidates(
        plan,
        per_role_limit=100,
        identity_limit=1000,
    )

    print(f"\nUnique candidate photos: {len(candidates)}")

    coverage = Counter(
        role
        for candidate in candidates
        for role in candidate.role_scores
    )

    print("\nCandidates by role:")
    for role in plan.roles:
        print(f"  {role.name}: {coverage[role.name]}")

    # Add eligible solo portraits missed by semantic search,
    # then validate the complete candidate pool.
    connection = connect(DATABASE_PATH)

    try:
        add_eligible_portrait_candidates(
            connection,
            candidates,
            IDENTITY_PATH,
        )

        evidence = load_portrait_evidence(
            connection,
            IDENTITY_PATH,
            {
                candidate.image_id
                for candidate in candidates
            },
        )
    finally:
        connection.close()

    eligible_before = sum(
        "main_portrait" in candidate.role_scores
        for candidate in candidates
    )

    filter_main_portrait_candidates(
        candidates,
        evidence,
    )

    # Exclude user-confirmed incorrect photos.
    candidates[:] = [
        candidate
        for candidate in candidates
        if candidate.image_id not in REJECTED_IDENTITY_IMAGE_IDS
    ]

    # Every selected photo must contain a detected face
    # matching the saved identity.
    #
    # The main_portrait role has already been checked
    # separately for exactly one detected face.
    for candidate in candidates:
        portrait_evidence = evidence.get(candidate.image_id)

        if (
            portrait_evidence is None
            or not portrait_evidence.target_identity_present
        ):
            candidate.role_scores.clear()
            candidate.role_ranks.clear()
            candidate.search_results.clear()

    eligible_after = sum(
        "main_portrait" in candidate.role_scores
        for candidate in candidates
    )

    print(
        f"\nPortrait candidates: "
        f"{eligible_before} before filtering, "
        f"{eligible_after} after filtering"
    )

    selections = select_collection(
        plan,
        candidates,
    )

    print(f"\nSelected {len(selections)} photos:\n")

    for index, selection in enumerate(
        selections,
        start=1,
    ):
        print(f"{index}. {selection.role}")
        print(f"   Image ID: {selection.image_id}")
        print(f"   Score: {selection.score:.3f}")
        print(f"   Path: {selection.path}")
        print()

    missing_roles = {
        role.name
        for role in plan.roles
    } - {
        selection.role
        for selection in selections
    }

    if missing_roles:
        print(
            "Unfilled roles:",
            ", ".join(sorted(missing_roles)),
        )
    else:
        print("All 6 collection roles filled.")


if __name__ == "__main__":
    main()
