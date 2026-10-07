from pathlib import Path
import sqlite3
import time

from combined_search import search_me_semantic


DATABASE_PATH = Path(
    ".recall/recall.db"
)

IDENTITY_PATH = Path(
    ".recall/identities/me.npy"
)


QUERIES = [
    "with people",
    "outside",
    "wearing a suit",
    "at night",
    "holding a camera",
]


def main() -> None:
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:
        print(
            "--- Recall Evaluation ---"
        )

        for query in QUERIES:
            start_time = time.perf_counter()

            results = search_me_semantic(
                connection,
                query,
                IDENTITY_PATH,
                identity_limit=100,
                limit=10,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f'\nQuery: "{query}"'
            )

            print(
                f"Results: {len(results)}"
            )

            print(
                f"Search time: "
                f"{elapsed:.3f}s"
            )

    finally:
        connection.close()


if __name__ == "__main__":
    main()