from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

from database import (
    get_image_id,
    needs_processing,
    record_processing_error,
    update_image_metadata,
)
register_heif_opener()


def read_image_metadata(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size

        return {
            "width": width,
            "height": height,
            "format": image.format,
        }


def process_images(
    connection,
    images: list[Path],
) -> dict[str, int]:

    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }

    thumbnail_directory = Path(".recall") / "thumbnails"

    for image in images:

        if not needs_processing(connection, image):
            stats["skipped"] += 1
            continue

        try:
            # Read dimensions/format
            metadata = read_image_metadata(image)

            # Get this image's SQLite ID
            image_id = get_image_id(
                connection,
                image,
            )

            if image_id is None:
                raise RuntimeError(
                    f"Image is not indexed: {image}"
                )

            # Example:
            # .recall/thumbnails/42.jpg
            thumbnail_path = (
                thumbnail_directory
                / f"{image_id}.jpg"
            )

            # Generate thumbnail
            generate_thumbnail(
                image,
                thumbnail_path,
            )

            # Store metadata + thumbnail location
            update_image_metadata(
                connection,
                image,
                metadata,
                thumbnail_path,
            )

            stats["processed"] += 1

        except Exception as error:
            record_processing_error(
                connection,
                image,
                str(error),
            )

            stats["failed"] += 1

    connection.commit()

    return stats

def generate_thumbnail(
    image_path: Path,
    thumbnail_path: Path,
    size: tuple[int, int] = (512, 512),
) -> None:

    thumbnail_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Image.open(image_path) as image:
        image.thumbnail(size)

        # JPEG can't save RGBA directly
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        image.save(
            thumbnail_path,
            format="JPEG",
            quality=85,
        )