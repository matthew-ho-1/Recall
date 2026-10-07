from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from pillow_heif import register_heif_opener
from database import (
    delete_faces_for_image,
    insert_face,
    mark_faces_processed,
    needs_face_processing,
    update_face_embedding,
)


register_heif_opener()

def load_face_model() -> FaceAnalysis:
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    return app

def detect_faces(
    app: FaceAnalysis,
    image_path: Path,
):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = np.array(image)

    # InsightFace/OpenCV expects BGR rather than RGB.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    return app.get(image)

def get_face_embedding(
    face,
) -> np.ndarray:

    embedding = face.embedding

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        raise ValueError(
            "Face embedding has zero norm"
        )

    return embedding / norm

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        np.dot(a, b)
    )

def get_first_face_embedding(
    app: FaceAnalysis,
    path: Path,
) -> np.ndarray:

    faces = detect_faces(
        app,
        path,
    )

    if not faces:
        raise ValueError(
            f"No face detected in {path}"
        )

    return get_face_embedding(
        faces[0]
    )

def save_face_embedding(
    embedding: np.ndarray,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        embedding,
    )
    
def load_face_embedding(
    path: Path,
) -> np.ndarray:
    return np.load(
        path
    )


def process_image_faces(
    connection,
    app: FaceAnalysis,
    image_path: Path,
) -> int:

    if not needs_face_processing(
        connection,
        image_path,
    ):
        return 0

    cursor = connection.execute(
        """
        SELECT id
        FROM images
        WHERE path = ?
        """,
        (str(image_path.resolve()),),
    )

    row = cursor.fetchone()

    if row is None:
        raise ValueError(
            f"Image is not indexed: {image_path}"
        )

    image_id = row[0]

    # Remove stale face data before regenerating it.
    delete_faces_for_image(
        connection,
        image_id,
    )

    faces = detect_faces(
        app,
        image_path,
    )

    for face in faces:
        embedding = get_face_embedding(
            face
        )

        face_id = insert_face(
            connection,
            image_id,
            face.bbox,
        )

        embedding_path = Path(
            ".recall/faces"
        ) / f"{face_id}.npy"

        save_face_embedding(
            embedding,
            embedding_path,
        )

        update_face_embedding(
            connection,
            face_id,
            embedding_path,
        )

    mark_faces_processed(
        connection,
        image_path,
    )

    connection.commit()

    return len(faces)

def process_faces(
    connection,
    images: list[Path],
) -> dict:
    stats = {
        "processed": 0,
        "skipped": 0,
        "faces_detected": 0,
        "failed": 0,
    }

    app = load_face_model()

    for image_path in images:
        try:
            if not needs_face_processing(
                connection,
                image_path,
            ):
                stats["skipped"] += 1
                continue

            face_count = process_image_faces(
                connection,
                app,
                image_path,
            )

            stats["processed"] += 1
            stats["faces_detected"] += face_count

        except Exception as error:
            connection.rollback()

            stats["failed"] += 1

            print(
                f"Failed to process faces for "
                f"{image_path}: {error}"
            )

    return stats