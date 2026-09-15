import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/clip/multimodal.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CLASSIFIER
# ============================================================

class Classifier(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.network(x)


# CLIP produces 512-dimensional text + 512-dimensional image
classifier = Classifier(1024).to(device)

classifier.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

classifier.eval()


# ============================================================
# LOAD CLIP
# ============================================================

print("Loading CLIP...")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_model.eval()

for param in clip_model.parameters():
    param.requires_grad = False


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(image_source):
 
    try:
 
        # Already a PIL Image -- e.g. Streamlit's app.py does
        # Image.open(uploaded_file) itself before calling predict().
        # Nothing to load, just make sure it's RGB.
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
 
        # A file-like object (Streamlit's raw UploadedFile, BytesIO, etc.)
        if hasattr(image_source, "read"):
            return Image.open(image_source).convert("RGB")
 
        # A URL
        if image_source.startswith("http"):
 
            response = requests.get(
                image_source,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"}
            )
 
            response.raise_for_status()
 
            return Image.open(
                BytesIO(response.content)
            ).convert("RGB")
 
        # A local file path
        else:
 
            return Image.open(
                image_source
            ).convert("RGB")
 
    except Exception as e:
 
        raise RuntimeError(
            f"Could not load image: {e}"
        )

# ============================================================
# PREDICTION
# ============================================================

def predict(text, image_path_or_url):

    image = load_image(
        image_path_or_url
    )

    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        text_output = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        image_output = clip_model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

    # Handle different Transformers versions
    if isinstance(text_output, torch.Tensor):
        text_features = text_output
    elif hasattr(text_output, "pooler_output"):
        text_features = text_output.pooler_output
    else:
        text_features = text_output.last_hidden_state[:, 0, :]

    if isinstance(image_output, torch.Tensor):
        image_features = image_output
    elif hasattr(image_output, "pooler_output"):
        image_features = image_output.pooler_output
    else:
        image_features = image_output.last_hidden_state[:, 0, :]

    # Normalize exactly as during training
    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-12)

    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-12)

    # Combine text + image
    multimodal_features = torch.cat(
        [text_features, image_features],
        dim=1
    )

    with torch.no_grad():

        output = classifier(
            multimodal_features
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

    label = "REAL" if prediction == 1 else "FAKE"

    confidence = probabilities[
        prediction
    ].item()

    return label, confidence


# ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("MULTIMODAL FAKE NEWS PREDICTOR")
    print("=" * 60)

    text = input("\nEnter text/title: ").strip()

    image = input(
        "Enter image path or image URL: "
    ).strip()

    label, confidence = predict(
        text,
        image
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print("Prediction :", label)
    print(
        "Confidence : "
        f"{confidence * 100:.2f}%"
    )