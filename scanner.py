import argparse
from collections import Counter
from pathlib import Path

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
        description="Recursively discover images in a directory."
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing photos",
    )

    args = parser.parse_args()

    if not args.directory.exists():
        raise SystemExit(
            f"Directory does not exist: {args.directory}"
        )

    if not args.directory.is_dir():
        raise SystemExit(
            f"Not a directory: {args.directory}"
        )

    print(f"Scanning {args.directory}...")

    # 1. Find all the images
    images = discover_images(args.directory)

    # 2. Count them by extension
    extension_counts = Counter(
        image.suffix.lower()
        for image in images
    )

    # 3. Print the results
    print(f"\nFound {len(images)} images:\n")

    for extension, count in sorted(extension_counts.items()):
        print(f"{extension:<6} {count:>5}")


if __name__ == "__main__":
    main()