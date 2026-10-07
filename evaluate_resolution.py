from pathlib import Path

from database import (
    connect,
    get_processed_images,
)
from quality import calculate_resolution


DATABASE_PATH = Path(".recall/recall.db")


def main() -> None:
    connection = connect(
        DATABASE_PATH
    )

    try:
        images = get_processed_images(
            connection
        )

        results = []

        for (
            image_id,
            image_path,
            thumbnail_path,
        ) in images:
            path = Path(
                image_path
            )

            try:
                resolution = (
                    calculate_resolution(
                        path
                    )
                )
            except Exception as error:
                print(
                    f"Failed [{image_id}] "
                    f"{path.name}: {error}"
                )
                continue

            results.append(
                {
                    "image_id": image_id,
                    "path": path,
                    **resolution,
                }
            )

        results.sort(
            key=lambda result:
                result["megapixels"]
        )

        print()
        print("Lowest resolution")
        print("-----------------")

        for result in results[:10]:
            print(
                f"{result['width']:5d} x "
                f"{result['height']:<5d}  "
                f"{result['megapixels']:6.2f} MP  "
                f"[{result['image_id']}]  "
                f"{result['path'].name}"
            )

        print()
        print("Highest resolution")
        print("------------------")

        for result in reversed(
            results[-10:]
        ):
            print(
                f"{result['width']:5d} x "
                f"{result['height']:<5d}  "
                f"{result['megapixels']:6.2f} MP  "
                f"[{result['image_id']}]  "
                f"{result['path'].name}"
            )

    finally:
        connection.close()


if __name__ == "__main__":
    main()