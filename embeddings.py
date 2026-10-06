from pathlib import Path

import open_clip
import torch
from PIL import Image
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

if __name__ == "__main__":
    model, preprocess, tokenizer = load_model()

    embedding = embed_image(
        model,
        preprocess,
        Path(".recall/thumbnails/1.jpg"),
    )

    print(embedding)
    print("Shape:", embedding.shape)