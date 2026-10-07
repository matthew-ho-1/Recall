from pathlib import Path
import sqlite3

from combined_search import search_me_semantic
from database import (
    get_relevance_judgment,
    save_relevance_judgment,
)


DATABASE_PATH = Path(
    ".recall/recall.db"
)

IDENTITY_PATH = Path(
    ".recall/identities/me.npy"
)


def evaluate_query(
    connection: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> None:

    # --------------------------------------------------
    # Run combined identity + semantic search
    # --------------------------------------------------

    results = search_me_semantic(
        connection,
        query,
        IDENTITY_PATH,
        identity_limit=100,
        limit=limit,
    )

    relevant_count = 0

    print(
        f'\n--- Evaluating: "{query}" ---'
    )

    # --------------------------------------------------
    # Evaluate each result
    # --------------------------------------------------

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{rank}. "
            f"{result['path']}"
        )

        print(
            f"   Semantic: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"   Identity: "
            f"{result['identity_score']:.4f}"
        )

        # ----------------------------------------------
        # Check for an existing saved judgment
        # ----------------------------------------------

        existing_judgment = (
            get_relevance_judgment(
                connection,
                query,
                result["image_id"],
            )
        )

        if existing_judgment is not None:
            relevant = existing_judgment

            print(
                "   Saved judgment:",
                (
                    "relevant"
                    if relevant
                    else "not relevant"
                ),
            )

        # ----------------------------------------------
        # Otherwise ask the user
        # ----------------------------------------------

        else:
            while True:
                answer = input(
                    "   Relevant? [y/n]: "
                ).strip().lower()

                if answer in {
                    "y",
                    "n",
                }:
                    break

                print(
                    "   Please enter y or n."
                )

            relevant = (
                answer == "y"
            )

            save_relevance_judgment(
                connection,
                query,
                result["image_id"],
                relevant,
            )

        # ----------------------------------------------
        # Count relevant results
        # ----------------------------------------------

        if relevant:
            relevant_count += 1

    # --------------------------------------------------
    # Calculate Precision@K
    # --------------------------------------------------

    if not results:
        precision = 0.0
    else:
        precision = (
            relevant_count
            / len(results)
        )

    # --------------------------------------------------
    # Print evaluation
    # --------------------------------------------------

    print(
        "\n--- Results ---"
    )

    print(
        f"Relevant: "
        f"{relevant_count}/"
        f"{len(results)}"
    )

    print(
        f"Precision@{len(results)}: "
        f"{precision:.3f}"
    )


def main() -> None:
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:
        query = input(
            "Query: "
        ).strip()

        evaluate_query(
            connection,
            query,
            limit=10,
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()