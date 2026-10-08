
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


register_heif_opener()


def compute_phash(image_path: Path) -> int:
    """
    Compute a 63-bit perceptual hash.

    Supports image formats decoded by Pillow,
    including HEIC/HEIF through pillow-heif.
    """
    image_path = Path(image_path)

    try:
        with Image.open(image_path) as image:
            # Respect EXIF orientation before hashing.
            image = ImageOps.exif_transpose(image)

            # Convert to grayscale.
            grayscale = image.convert("L")

            # Resize to the standard pHash input size.
            resized = grayscale.resize(
                (32, 32),
                Image.Resampling.LANCZOS,
            )

            pixels = np.asarray(
                resized,
                dtype=np.float32,
            )

    except Exception as error:
        raise ValueError(
            f"Could not read image: {image_path}"
        ) from error

    # Discrete cosine transform.
    dct = cv2.dct(pixels)

    # Keep the low-frequency coefficients.
    low_frequency = dct[:8, :8].flatten()

    # Exclude the DC coefficient.
    coefficients = low_frequency[1:]

    median = np.median(coefficients)

    bits = coefficients > median

    value = 0

    for bit in bits:
        value = (value << 1) | int(bit)

    return value


def hamming_distance(
    hash_a: int,
    hash_b: int,
) -> int:
    """
    Count differing bits between two hashes.
    """
    return (hash_a ^ hash_b).bit_count()
