from pathlib import Path

import numpy as np

import shutil
import sqlite3

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

def set_identity_from_indexed_photo(
    image_id: int,
    *,
    database_path: Path = Path(".recall/recall.db"),
    identity_path: Path = Path(".recall/identities/me.npy"),
) -> Path:
    """
    Set the target identity using an indexed photo
    containing exactly one detected face.

    Back up the existing identity without overwriting
    any previous backups.
    """
    if not database_path.is_file():
        raise FileNotFoundError(
            f"Recall database not found: {database_path}"
        )

    connection = sqlite3.connect(database_path)

    try:
        image = connection.execute(
            "SELECT path FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()

        if image is None:
            raise ValueError(
                f"Image ID {image_id} was not found."
            )

        rows = connection.execute(
            """
            SELECT embedding_path
            FROM faces
            WHERE image_id = ?
            ORDER BY id
            """,
            (image_id,),
        ).fetchall()

    finally:
        connection.close()

    if len(rows) != 1:
        raise ValueError(
            f"Image ID {image_id} must have exactly "
            f"one detected face; found {len(rows)}."
        )

    face_path = Path(rows[0][0])

    if not face_path.is_file():
        raise FileNotFoundError(
            f"Face embedding not found: {face_path}"
        )

    embedding = np.load(face_path).reshape(-1)
    norm = np.linalg.norm(embedding)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid face embedding.")

    embedding = embedding / norm

    identity_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Back up the previous identity using a unique filename.
    if identity_path.exists():
        backup_number = 1

        while True:
            backup_path = identity_path.with_name(
                f"{identity_path.stem}.backup-{backup_number}.npy"
            )

            if not backup_path.exists():
                break

            backup_number += 1

        shutil.copy2(identity_path, backup_path)
        print(f"Previous identity backed up: {backup_path}")

    # Save the new identity.
    np.save(identity_path, embedding)

    print(f"Identity set from image ID {image_id}")
    print(f"Source photo: {image[0]}")
    print(f"Saved identity: {identity_path}")

    return identity_path

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