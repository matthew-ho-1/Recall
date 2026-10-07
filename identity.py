from pathlib import Path

import numpy as np

from faces import (
    detect_faces,
    get_face_embedding,
    load_face_model,
)

def build_identity_embedding(
    app,
    reference_images: list[Path],
) -> np.ndarray:

    embeddings = []

    for image_path in reference_images:
        faces = detect_faces(
            app,
            image_path,
        )

        if len(faces) != 1:
            raise ValueError(
                f"Expected exactly one face in "
                f"{image_path}, found {len(faces)}"
            )

        embedding = get_face_embedding(
            faces[0]
        )

        embeddings.append(
            embedding
        )

    identity = np.mean(
        embeddings,
        axis=0,
    )

    identity = identity / np.linalg.norm(
        identity
    )

    return identity

def get_single_face_embedding(
    app,
    image_path: Path,
) -> np.ndarray:

    faces = detect_faces(
        app,
        image_path,
    )

    if len(faces) != 1:
        raise ValueError(
            f"Expected exactly one face in "
            f"{image_path}, found {len(faces)}"
        )

    return get_face_embedding(
        faces[0]
    )

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        np.dot(a, b)
    )

def save_identity_embedding(
    identity: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        identity,
    )

def load_identity_embedding(
    path: Path,
) -> np.ndarray:
    identity = np.load(
        path
    )

    return identity

"""
   Path(
           "D:\\Pictures\\olivia\\IMG_5861.jpg"
        ),
        Path(
           "D:\\Pictures\\olivia\\IMG_4695.HEIC"
        ),
        Path(
            "D:\\Pictures\\olivia\\IMG_6255.jpg"
        ),
"""

if __name__ == "__main__":
    app = load_face_model()

    # Use the same reference photos that worked in 15.2.
    reference_images = [
        Path(
            "D:\\Pictures\\olivia\\IMG_5861.jpg"
        ),
        Path(
            "D:\\Pictures\\olivia\\IMG_4695.HEIC"
        ),
        Path(
            "D:\\Pictures\\olivia\\IMG_6255.jpg"
        ),
    ]

    # Build "me".
    identity = build_identity_embedding(
        app,
        reference_images,
    )

    # Save it.
    identity_path = Path(
        ".recall/identities/me.npy"
    )

    save_identity_embedding(
        identity,
        identity_path,
    )

    print(
        "Saved identity:",
        identity_path,
    )

    print(
        "File exists:",
        identity_path.exists(),
    )

    # Load it back.
    loaded_identity = load_identity_embedding(
        identity_path
    )

    print(
        "Loaded shape:",
        loaded_identity.shape,
    )

    print(
        "Loaded norm:",
        np.linalg.norm(
            loaded_identity
        ),
    )

    print(
        "Same embedding:",
        np.allclose(
            identity,
            loaded_identity,
        ),
    )