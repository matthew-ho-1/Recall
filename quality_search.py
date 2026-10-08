
import argparse
import sqlite3
from pathlib import Path

import numpy as np

from combined_search import search_me_semantic
from database import (
    connect,
    get_cached_quality_measurements,
    get_searchable_faces,
    save_quality_measurements,
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
from quality_reranker import rerank_by_quality


DATABASE_PATH = Path(".recall/recall.db")
IDENTITY_PATH = Path(".recall/identities/me.npy")


def load_normalized_embedding(path: Path) -> np.ndarray:
    embedding = np.load(path)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError(f"Zero-norm embedding: {path}")

    return embedding / norm



def get_best_faces_by_image(
    connection: sqlite3.Connection,
    identity_path: Path,
    candidate_ids: set[int],
) -> dict[int, dict]:
    identity_embedding = load_normalized_embedding(identity_path)
    best_faces = {}

    for (
        face_id,
        image_id,
        image_path,
        x1,
        y1,
        x2,
        y2,
        embedding_path,
    ) in get_searchable_faces(connection):
        if image_id not in candidate_ids:
            continue

        face_embedding = load_normalized_embedding(
            Path(embedding_path)
        )
        similarity = float(
            np.dot(identity_embedding, face_embedding)
        )

        current = best_faces.get(image_id)

        if current is None or similarity > current["similarity"]:
            best_faces[image_id] = {
                "face_id": face_id,
                "similarity": similarity,
                "bbox": (x1, y1, x2, y2),
            }

    return {
        image_id: {
            "face_id": face["face_id"],
            "bbox": face["bbox"],
        }
        for image_id, face in best_faces.items()
    }




def search_me_with_quality(
    connection: sqlite3.Connection,
    query: str,
    identity_path: Path,
    *,
    identity_limit: int = 100,
    candidate_limit: int = 10,
    rerank: bool = False,
) -> list[dict]:
    results = search_me_semantic(
        connection,
        query,
        identity_path,
        identity_limit=identity_limit,
        limit=candidate_limit,
    )

    candidate_ids = {
        result["image_id"]
        for result in results
    }

    best_faces = get_best_faces_by_image(
        connection,
        identity_path,
        candidate_ids,
    )

    enriched_results = []

    for result in results:
        image_id = result["image_id"]
        image_path = Path(result["path"])
        face_info = best_faces.get(image_id)

        stat = image_path.stat()

        measurements = None

        # Try to reuse measurements for this exact
        # image version and selected face.
        if face_info is not None:
            measurements = get_cached_quality_measurements(
                connection,
                image_id,
                face_info["face_id"],
                stat.st_size,
                stat.st_mtime,
            )

        # Cache miss: calculate measurements as before.
        if measurements is None:
            whole_exposure = calculate_exposure(image_path)
            resolution = calculate_resolution(image_path)

            face_sharpness = None
            face_mean_brightness = None
            face_prominence = None

            if face_info is not None:
                bbox = face_info["bbox"]

                face_sharpness = calculate_face_sharpness(
                    image_path,
                    bbox,
                )

                face_exposure = calculate_region_exposure(
                    image_path,
                    bbox,
                )
                face_mean_brightness = face_exposure[
                    "mean_brightness"
                ]

                prominence = calculate_face_prominence(
                    image_path,
                    bbox,
                )
                face_prominence = prominence[
                    "face_area_fraction"
                ]

            measurements = {
                "face_sharpness": face_sharpness,
                "whole_mean_brightness": whole_exposure[
                    "mean_brightness"
                ],
                "face_mean_brightness": face_mean_brightness,
                "megapixels": resolution["megapixels"],
                "face_prominence": face_prominence,
            }

            # Identity results normally have a face.
            # If one is missing, calculate quality but
            # skip the face-specific cache.
            if face_info is not None:
                save_quality_measurements(
                    connection,
                    image_id,
                    face_info["face_id"],
                    stat.st_size,
                    stat.st_mtime,
                    measurements,
                )

        quality = evaluate_quality(**measurements)

        enriched_results.append(
            {
                **result,
                "quality_tier": quality_tier(quality),
                "quality_evidence": quality,
            }
        )

    # Persist new measurements for future searches.
    connection.commit()

    if rerank:
        return rerank_by_quality(enriched_results)

    return enriched_results



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search Recall using identity, semantic "
            "relevance, and technical photo quality."
        )
    )

    parser.add_argument(
        "query",
        help="Natural-language search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of results to display",
    )

    parser.add_argument(
        "--identity-limit",
        type=int,
        default=100,
        help="Number of identity candidates to search",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=30,
        help=(
            "Number of semantic candidates to "
            "evaluate when quality reranking is enabled"
        ),
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply quality-aware reranking",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.identity_limit < 1:
        parser.error("--identity-limit must be at least 1")

    if args.pool_size < 1:
        parser.error("--pool-size must be at least 1")

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(DATABASE_PATH)

    if not IDENTITY_PATH.exists():
        raise FileNotFoundError(IDENTITY_PATH)

    # --------------------------------------------------
    # 5. Determine how many candidates to evaluate
    # --------------------------------------------------

    if args.rerank:
        candidate_limit = max(
            args.pool_size,
            args.limit,
        )
    else:
        candidate_limit = args.limit

    # --------------------------------------------------
    # 6. Run the search
    # --------------------------------------------------

    connection = connect(DATABASE_PATH)

    try:
        results = search_me_with_quality(
            connection,
            args.query,
            IDENTITY_PATH,
            identity_limit=args.identity_limit,
            candidate_limit=candidate_limit,
            rerank=args.rerank,
        )
    finally:
        connection.close()

    # --------------------------------------------------
    # 7. Display the final top results
    # --------------------------------------------------

    mode = (
        "Quality-aware ranking"
        if args.rerank
        else "Semantic-only ranking"
    )

    print(f'\n{mode}: "{args.query}"')
    print("-" * 45)

    if args.rerank:
        print(
            f"Candidate pool: {len(results)}"
        )

    final_results = results[:args.limit]

    if not final_results:
        print("No matching images found.")
        return

    for rank, result in enumerate(
        final_results,
        start=1,
    ):
        evidence = result["quality_evidence"]

        defects = []

        if evidence.blur_evidence:
            defects.append("blur")

        if evidence.underexposure_evidence:
            defects.append("underexposed")

        if evidence.overexposure_evidence:
            defects.append("overexposed")

        defect_text = (
            ", ".join(defects)
            if defects
            else "none detected"
        )

        print(
            f"\n{rank}. "
            f"semantic={result['semantic_score']:.4f} "
            f"identity={result['identity_score']:.4f} "
            f"tier={result['quality_tier']}"
        )

        print(f"   {result['path']}")
        print(f"   Defects: {defect_text}")


if __name__ == "__main__":
    main()
