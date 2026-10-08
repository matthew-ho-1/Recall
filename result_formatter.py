
"""
Formatting utilities for Recall search results.

Converts internal search result dictionaries into
JSON-compatible data without exposing raw model objects.
"""

from pathlib import Path


def format_result(result: dict, rank: int) -> dict:
    """Convert one search result into JSON-compatible data."""

    evidence = result["quality_evidence"]

    defects = []

    if evidence.blur_evidence:
        defects.append("blur")

    if evidence.underexposure_evidence:
        defects.append("underexposed")

    if evidence.overexposure_evidence:
        defects.append("overexposed")

    return {
        "rank": rank,
        "image_id": result["image_id"],
        "path": str(Path(result["path"])),
        "semantic_score": float(result["semantic_score"]),
        "identity_score": float(result["identity_score"]),
        "quality_tier": int(result["quality_tier"]),
        "defects": defects,
    }


def format_results(
    results: list[dict],
    *,
    query: str,
    rerank: bool,
    selection: str,
) -> dict:
    """Build a JSON-compatible search response."""

    return {
        "query": query,
        "ranking": (
            "quality-aware"
            if rerank
            else "semantic-only"
        ),
        "selection": selection,
        "count": len(results),
        "results": [
            format_result(result, rank)
            for rank, result in enumerate(results, start=1)
        ],
    }
