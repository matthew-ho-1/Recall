
def rerank_by_quality(
    results: list[dict],
    semantic_tolerance: float = 0.005,
) -> list[dict]:
    """
    Prefer technically cleaner photos when semantic
    relevance is sufficiently similar.

    At each position:
    1. Find the highest remaining semantic score.
    2. Consider candidates within semantic_tolerance.
    3. Prefer the lowest quality tier.
    4. Break quality ties by semantic relevance.

    Does not modify the input list.
    """

    if semantic_tolerance < 0:
        raise ValueError(
            "semantic_tolerance must be nonnegative"
        )

    remaining = list(results)
    reranked = []

    while remaining:
        best_semantic = max(
            result["semantic_score"]
            for result in remaining
        )

        eligible = [
            (index, result)
            for index, result in enumerate(remaining)
            if (
                best_semantic - result["semantic_score"]
                <= semantic_tolerance
            )
        ]

        selected_index, _ = min(
            eligible,
            key=lambda item: (
                item[1]["quality_tier"],
                -item[1]["semantic_score"],
                item[0],
            ),
        )

        reranked.append(
            remaining.pop(selected_index)
        )

    return reranked


if __name__ == "__main__":
    sample = [
        {
            "path": "photo_A.jpg",
            "semantic_score": 0.2581,
            "quality_tier": 0,
        },
        {
            "path": "photo_B.jpg",
            "semantic_score": 0.2356,
            "quality_tier": 1,
        },
        {
            "path": "photo_C.jpg",
            "semantic_score": 0.2316,
            "quality_tier": 0,
        },
    ]

    ranked = rerank_by_quality(sample)

    assert [
        result["path"]
        for result in ranked
    ] == [
        "photo_A.jpg",
        "photo_C.jpg",
        "photo_B.jpg",
    ]

    assert sample[1]["path"] == "photo_B.jpg"

    print("Quality reranker smoke test passed.")
    print()

    for index, result in enumerate(ranked, start=1):
        print(
            f"{index}. {result['path']} "
            f"| semantic={result['semantic_score']:.4f} "
            f"| tier={result['quality_tier']}"
        )
