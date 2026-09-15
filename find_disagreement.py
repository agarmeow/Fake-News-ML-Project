import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "models/clip"
TEST_FILE = "data/multimodal_clean/test.tsv"

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


# ============================================================
# LOAD SAVED EMBEDDINGS
# ============================================================

test_text = np.load(
    f"{DATA_DIR}/test_text.npy"
)

test_image = np.load(
    f"{DATA_DIR}/test_image.npy"
)

test_labels = np.load(
    f"{DATA_DIR}/test_labels.npy"
)

print("Usable test embeddings:", len(test_labels))


# ============================================================
# LOAD MODELS
# ============================================================

text_model = Classifier(
    test_text.shape[1]
).to(device)

image_model = Classifier(
    test_image.shape[1]
).to(device)

multimodal_model = Classifier(
    test_text.shape[1] + test_image.shape[1]
).to(device)


text_model.load_state_dict(
    torch.load(
        f"{DATA_DIR}/text_only.pt",
        map_location=device
    )
)

image_model.load_state_dict(
    torch.load(
        f"{DATA_DIR}/image_only.pt",
        map_location=device
    )
)

multimodal_model.load_state_dict(
    torch.load(
        f"{DATA_DIR}/multimodal.pt",
        map_location=device
    )
)

text_model.eval()
image_model.eval()
multimodal_model.eval()


# ============================================================
# GET PREDICTIONS
# ============================================================

with torch.no_grad():

    text_output = text_model(
        torch.tensor(
            test_text,
            dtype=torch.float32
        ).to(device)
    )

    image_output = image_model(
        torch.tensor(
            test_image,
            dtype=torch.float32
        ).to(device)
    )

    multimodal_input = np.concatenate(
        [test_text, test_image],
        axis=1
    )

    multimodal_output = multimodal_model(
        torch.tensor(
            multimodal_input,
            dtype=torch.float32
        ).to(device)
    )

text_predictions = torch.argmax(
    text_output,
    dim=1
).cpu().numpy()

image_predictions = torch.argmax(
    image_output,
    dim=1
).cpu().numpy()

multimodal_predictions = torch.argmax(
    multimodal_output,
    dim=1
).cpu().numpy()


# ============================================================
# FIND DISAGREEMENTS
# ============================================================

disagreement = (
    text_predictions != multimodal_predictions
)

indices = np.where(disagreement)[0]

print("\nText vs Multimodal disagreements:", len(indices))


# ============================================================
# LOAD ORIGINAL TEST DATA
# ============================================================

df = pd.read_csv(
    TEST_FILE,
    sep="\t"
)


# ============================================================
# MATCH SAVED EMBEDDINGS TO ORIGINAL TEST ROWS
#
# Some images failed during extraction, so the embeddings
# contain 1888 rows while the TSV contains 2000 rows.
#
# We regenerate ONLY text embeddings to identify the
# corresponding original rows. This is much faster than
# downloading the images again.
# ============================================================

print("\nLoading CLIP for row matching...")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_model.eval()

all_text_embeddings = []

BATCH_SIZE = 32

texts = df["clean_title"].astype(str).tolist()

print("Generating text embeddings for row matching...")

for start in range(0, len(texts), BATCH_SIZE):

    batch = texts[
        start:start + BATCH_SIZE
    ]

    inputs = processor(
        text=batch,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        output = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    if isinstance(output, torch.Tensor):
        features = output

    elif hasattr(output, "pooler_output"):
        features = output.pooler_output

    elif hasattr(output, "last_hidden_state"):
        features = output.last_hidden_state[:, 0, :]

    else:
        raise TypeError(
            f"Unexpected CLIP output: {type(output)}"
        )

    features = features / features.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-12)

    all_text_embeddings.append(
        features.cpu().numpy()
    )

all_text_embeddings = np.concatenate(
    all_text_embeddings,
    axis=0
)


# ============================================================
# MATCH EACH SAVED TEST EMBEDDING TO ORIGINAL ROW
# ============================================================

similarities = cosine_similarity(
    test_text,
    all_text_embeddings
)

matched_rows = np.argmax(
    similarities,
    axis=1
)

match_scores = np.max(
    similarities,
    axis=1
)


# ============================================================
# PRINT DISAGREEMENT EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("QUALITATIVE DISAGREEMENT EXAMPLES")
print("=" * 70)

shown = 0

for idx in indices:

    original_row = matched_rows[idx]
    similarity = match_scores[idx]

    # Require a very strong text-embedding match
    if similarity < 0.999:
        continue

    row = df.iloc[original_row]

    print("\n" + "-" * 70)

    print("Test embedding index:", idx)
    print("Original test row:", original_row)
    print("Text matching similarity:", round(float(similarity), 6))

    print("\nCLAIM / TITLE:")
    print(row["clean_title"])

    print("\nTRUE LABEL:")
    print(int(test_labels[idx]))

    print("\nTEXT-ONLY PREDICTION:")
    print(int(text_predictions[idx]))

    print("\nIMAGE-ONLY PREDICTION:")
    print(int(image_predictions[idx]))

    print("\nMULTIMODAL PREDICTION:")
    print(int(multimodal_predictions[idx]))

    print("\nIMAGE URL:")
    print(row["image_url"])

    shown += 1

    if shown >= 5:
        break


print("\n" + "=" * 70)
print(f"Displayed {shown} disagreement examples.")
print("=" * 70)