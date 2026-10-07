from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener


register_heif_opener()


def calculate_sharpness(image_path: Path) -> float:
    """
    Calculate the sharpness of an image using variance of the Laplacian.

    Higher values generally indicate a sharper image.

    This returns a raw measurement, not a normalized quality score.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(
        grayscale,
        cv2.CV_64F,
    )

    return float(laplacian.var())

def calculate_face_sharpness(
    image_path: Path,
    bbox: tuple[float, float, float, float],
    target_size: tuple[int, int] = (256, 256),
) -> float:
    """
    Calculate sharpness within a face region.

    The face region is resized to a standard size before
    calculating Laplacian variance so that face size has
    less influence on the result.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = np.array(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    height, width = image.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bounding box: {bbox}"
        )

    region = image[
        y1:y2,
        x1:x2,
    ]

    region_height, region_width = (
        region.shape[:2]
    )

    if (
        region_width > target_size[0]
        or region_height > target_size[1]
    ):
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    region = cv2.resize(
        region,
        target_size,
        interpolation=interpolation,
    )

    grayscale = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY,
    )

    laplacian = cv2.Laplacian(
        grayscale,
        cv2.CV_64F,
    )

    return float(
        laplacian.var()
    )

def calculate_exposure(
    image_path: Path,
) -> dict[str, float]:
    """
    Measure basic exposure characteristics of an image.

    Returns raw measurements rather than a combined
    exposure-quality score.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = np.array(image)

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    mean_brightness = float(
        grayscale.mean()
    )

    dark_fraction = float(
        np.mean(
            grayscale <= 25
        )
    )

    bright_fraction = float(
        np.mean(
            grayscale >= 230
        )
    )

    return {
        "mean_brightness": mean_brightness,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
    }

def calculate_region_exposure(
    image_path: Path,
    bbox: tuple[float, float, float, float],
) -> dict[str, float]:
    """
    Measure exposure characteristics inside a specific
    region of the original image.

    bbox uses original-image coordinates.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = np.array(image)

    height, width = image.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        min(width, int(x1)),
    )
    y1 = max(
        0,
        min(height, int(y1)),
    )
    x2 = max(
        0,
        min(width, int(x2)),
    )
    y2 = max(
        0,
        min(height, int(y2)),
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bounding box: {bbox}"
        )

    region = image[
        y1:y2,
        x1:x2,
    ]

    grayscale = cv2.cvtColor(
        region,
        cv2.COLOR_RGB2GRAY,
    )

    mean_brightness = float(
        grayscale.mean()
    )

    dark_fraction = float(
        np.mean(
            grayscale <= 25
        )
    )

    bright_fraction = float(
        np.mean(
            grayscale >= 230
        )
    )

    return {
        "mean_brightness": mean_brightness,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
    }

def calculate_resolution(
    image_path: Path,
) -> dict[str, int | float]:
    """
    Measure the native resolution of an image.

    Returns raw measurements rather than a
    combined resolution-quality score.
    """
    with Image.open(image_path) as image:
        width, height = image.size

    megapixels = (
        width * height
    ) / 1_000_000

    return {
        "width": width,
        "height": height,
        "megapixels": megapixels,
    }

def calculate_face_prominence(
    image_path: Path,
    bbox: tuple[float, float, float, float],
) -> dict[str, float]:
    """
    Measure how prominent a face is relative
    to the full image.

    bbox uses original-image coordinates.
    """
    with Image.open(image_path) as image:
        width, height = image.size

    x1, y1, x2, y2 = bbox

    x1 = max(
        0.0,
        min(float(width), x1),
    )
    y1 = max(
        0.0,
        min(float(height), y1),
    )
    x2 = max(
        0.0,
        min(float(width), x2),
    )
    y2 = max(
        0.0,
        min(float(height), y2),
    )

    face_width = max(
        0.0,
        x2 - x1,
    )
    face_height = max(
        0.0,
        y2 - y1,
    )

    image_area = (
        width * height
    )

    face_area = (
        face_width * face_height
    )

    return {
        "face_width_fraction":
            face_width / width,
        "face_height_fraction":
            face_height / height,
        "face_area_fraction":
            face_area / image_area,
    }