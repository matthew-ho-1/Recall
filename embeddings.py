import open_clip

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

if __name__ == "__main__":
    model, preprocess, tokenizer = load_model()

    print("Model loaded successfully")