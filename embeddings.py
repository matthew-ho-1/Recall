from pathlib import Path

import open_clip
import torch
from PIL import Image

import numpy as np

from database import (
    get_image_id,
    needs_embedding,
    update_image_embedding,
)


MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )

    tokenizer = open_clip.get_tokenizer(
        MODEL_NAME
    )

    model.eval()

    return model, preprocess, tokenizer


def embed_image(
    model,
    preprocess,
    image_path: Path,
) -> torch.Tensor:

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(
            image_tensor
        )

    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True,
    )

    return embedding.squeeze(0)


def embed_text(
    model,
    tokenizer,
    text: str,
) -> torch.Tensor:

    tokens = tokenizer([text])

    with torch.no_grad():
        embedding = model.encode_text(tokens)

    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True,
    )

    return embedding.squeeze(0)

def save_embedding(
    embedding: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    array = embedding.cpu().numpy()

    np.save(
        path,
        array,
    )

def generate_embeddings(
    connection,
    images: list[Path],
) -> dict[str, int]:

    stats = {
        "embedded": 0,
        "skipped": 0,
        "failed": 0,
    }

    print("Loading embedding model...")

    model, preprocess, _ = load_model()

    embedding_directory = (
        Path(".recall") / "embeddings"
    )

    for image in images:

        if not needs_embedding(
            connection,
            image,
        ):
            stats["skipped"] += 1
            continue

        try:
            image_id = get_image_id(
                connection,
                image,
            )

            if image_id is None:
                raise RuntimeError(
                    f"Image is not indexed: {image}"
                )

            thumbnail_path = (
                Path(".recall")
                / "thumbnails"
                / f"{image_id}.jpg"
            )

            if not thumbnail_path.exists():
                raise RuntimeError(
                    f"Thumbnail does not exist: "
                    f"{thumbnail_path}"
                )

            embedding = embed_image(
                model,
                preprocess,
                thumbnail_path,
            )

            embedding_path = (
                embedding_directory
                / f"{image_id}.npy"
            )

            save_embedding(
                embedding,
                embedding_path,
            )

            update_image_embedding(
                connection,
                image,
                embedding_path,
            )

            stats["embedded"] += 1

        except Exception as error:
            print(
                f"Failed to embed {image}: "
                f"{error}"
            )

            stats["failed"] += 1

    connection.commit()

    return stats