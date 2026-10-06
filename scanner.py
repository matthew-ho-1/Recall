import argparse
from collections import Counter
from pathlib import Path

from database import (
    connect,
    index_images,
    initialize_database,
    remove_deleted_images,
)

from processor import process_images


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".heif",
    ".webp",
}


def discover_images(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Recursively discover and index images."
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing photos",
    )

    args = parser.parse_args()

    # Validate directory
    if not args.directory.exists():
        raise SystemExit(
            f"Directory does not exist: {args.directory}"
        )

    if not args.directory.is_dir():
        raise SystemExit(
            f"Not a directory: {args.directory}"
        )

    # Discover images
    print(f"Scanning {args.directory}...")

    images = discover_images(args.directory)

    # Count images by extension
    extension_counts = Counter(
        image.suffix.lower()
        for image in images
    )

    print(f"\nFound {len(images)} images:\n")

    for extension, count in sorted(extension_counts.items()):
        print(f"{extension:<6} {count:>5}")

    # Open Recall database
    database_path = Path(".recall") / "recall.db"

    connection = connect(database_path)

    try:
        initialize_database(connection)

        print("\nIndexing images...")

        stats = index_images(
            connection,
            images,
        )

        deleted = remove_deleted_images(
            connection,
            images,
            args.directory,
        )

        print("\nProcessing images...")

        processing_stats = process_images(
            connection,
            images,
        )

        print("\nProcessing:")
        print(f"Processed: {processing_stats['processed']}")
        print(f"Skipped:   {processing_stats['skipped']}")
        print(f"Failed:    {processing_stats['failed']}")

    finally:
        connection.close()

    # Print indexing results
    indexed = stats["new"] + stats["modified"]

    print(f"\nDiscovered: {len(images)}")
    print(f"New:        {stats['new']}")
    print(f"Modified:   {stats['modified']}")
    print(f"Unchanged:  {stats['unchanged']}")
    print(f"Indexed:    {indexed}")
    print(f"Deleted:    {deleted}")

    print(f"\nDatabase: {database_path}")


if __name__ == "__main__":
    main()