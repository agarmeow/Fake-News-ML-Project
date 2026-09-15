import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
import requests
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 16

DATA_DIR = "data/multimodal_clean"
OUTPUT_DIR = "models/clip"

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    f"{DATA_DIR}/train.tsv",
    sep="\t"
)

val_df = pd.read_csv(
    f"{DATA_DIR}/validate.tsv",
    sep="\t"
)

test_df = pd.read_csv(
    f"{DATA_DIR}/test.tsv",
    sep="\t"
)

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# LOAD CLIP
# ============================================================

print("\nLoading CLIP...")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_model.to(device)
clip_model.eval()

# Freeze CLIP
for param in clip_model.parameters():
    param.requires_grad = False

print("CLIP loaded and frozen.")


# ============================================================
# DOWNLOAD IMAGE
# ============================================================

def download_image(url):

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if response.status_code != 200:
            return None

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        return image

    except Exception:
        return None


# ============================================================
# GET RAW CLIP EMBEDDING
# ============================================================

def get_embedding(output):

    # Some Transformers versions return a tensor.
    # Other versions may return a model output object.

    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]

    raise TypeError(
        f"Unexpected CLIP output type: {type(output)}"
    )


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

def extract_embeddings(df, split_name):

    text_embeddings = []
    image_embeddings = []
    labels = []

    texts = []
    images = []
    batch_labels = []

    failed_images = 0
    failed_batches = 0

    print(f"\nProcessing {split_name}...")

    progress = tqdm(
        total=len(df),
        desc=split_name
    )

    def process_batch():

        nonlocal failed_batches

        if len(images) == 0:
            return

        try:

            inputs = processor(
                text=texts,
                images=images,
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

            # Convert model output to actual tensors
            text_features = get_embedding(text_output)
            image_features = get_embedding(image_output)

            # Normalize
            text_features = (
                text_features /
                text_features.norm(
                    dim=-1,
                    keepdim=True
                ).clamp(min=1e-12)
            )

            image_features = (
                image_features /
                image_features.norm(
                    dim=-1,
                    keepdim=True
                ).clamp(min=1e-12)
            )

            text_embeddings.extend(
                text_features.cpu().numpy()
            )

            image_embeddings.extend(
                image_features.cpu().numpy()
            )

            labels.extend(batch_labels)

        except Exception as e:

            failed_batches += 1

            print(
                f"\nBatch failed: {e}"
            )

        finally:

            texts.clear()
            images.clear()
            batch_labels.clear()

    # --------------------------------------------------------
    # Process rows
    # --------------------------------------------------------

    for _, row in df.iterrows():

        text = str(row["clean_title"])

        image = download_image(
            row["image_url"]
        )

        if image is None:

            failed_images += 1
            progress.update(1)
            continue

        texts.append(text)
        images.append(image)
        batch_labels.append(
            int(row["2_way_label"])
        )

        progress.update(1)

        if len(images) >= BATCH_SIZE:
            process_batch()

    # Process final incomplete batch
    process_batch()

    progress.close()

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    text_embeddings = np.asarray(
        text_embeddings,
        dtype=np.float32
    )

    image_embeddings = np.asarray(
        image_embeddings,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    print(
        f"\n{split_name} results:"
    )

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Failed image downloads: {failed_images}"
    )

    print(
        f"Failed batches: {failed_batches}"
    )

    print(
        f"Usable samples: {len(labels)}"
    )

    print(
        f"Text embedding shape: {text_embeddings.shape}"
    )

    print(
        f"Image embedding shape: {image_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(labels) == 0:
        raise RuntimeError(
            f"No usable samples were produced for {split_name}."
        )

    if len(text_embeddings) != len(image_embeddings):
        raise RuntimeError(
            f"Text/image embedding count mismatch in {split_name}."
        )

    if len(text_embeddings) != len(labels):
        raise RuntimeError(
            f"Embedding/label count mismatch in {split_name}."
        )

    if failed_batches > 0:
        raise RuntimeError(
            f"{failed_batches} batches failed in {split_name}. "
            "Stopping instead of silently producing incomplete data."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        f"{OUTPUT_DIR}/{split_name}_text.npy",
        text_embeddings
    )

    np.save(
        f"{OUTPUT_DIR}/{split_name}_image.npy",
        image_embeddings
    )

    np.save(
        f"{OUTPUT_DIR}/{split_name}_labels.npy",
        labels
    )

    print(
        f"Saved {split_name} embeddings."
    )

    return (
        text_embeddings,
        image_embeddings,
        labels
    )


# ============================================================
# EXTRACT ALL EMBEDDINGS
# ============================================================

train_text, train_image, train_labels = extract_embeddings(
    train_df,
    "train"
)

val_text, val_image, val_labels = extract_embeddings(
    val_df,
    "validate"
)

test_text, test_image, test_labels = extract_embeddings(
    test_df,
    "test"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)

print("\nTrain:")
print(" Text :", train_text.shape)
print(" Image:", train_image.shape)
print(" Labels:", train_labels.shape)

print("\nValidation:")
print(" Text :", val_text.shape)
print(" Image:", val_image.shape)
print(" Labels:", val_labels.shape)

print("\nTest:")
print(" Text :", test_text.shape)
print(" Image:", test_image.shape)
print(" Labels:", test_labels.shape)

print("\nSaved to:")
print(OUTPUT_DIR)