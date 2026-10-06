from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from pillow_heif import register_heif_opener


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

if __name__ == "__main__":
    app = load_face_model()

    a = get_first_face_embedding(
        app,
        Path("IMG_5861.jpg"),
    )

    b = get_first_face_embedding(
        app,
        Path("IMG_4695.HEIC"),
    )

    c = get_first_face_embedding(
        app,
        Path("105_4462.jpg"),
    )

    print(
        "Same person:",
        cosine_similarity(a, b),
    )

    print(
        "Different person:",
        cosine_similarity(a, c),
    )