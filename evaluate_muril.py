import pandas as pd
import numpy as np
import torch

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path("models/muril_classifier")
TEST_FILE = Path("data/processed/splits/test.csv")

MAX_LENGTH = 256
BATCH_SIZE = 8


# ============================================================
# Dataset
# ============================================================

class FakeNewsDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=256):

        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        title = str(row["title"]) if pd.notna(row["title"]) else ""
        body = str(row["body"]) if pd.notna(row["body"]) else ""

        text = title + " " + body

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return item


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Load test data
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_FILE)

print(f"Test samples: {len(test_df)}")


# ============================================================
# Load trained MuRIL
# ============================================================

print("\nLoading trained MuRIL model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Create DataLoader
# ============================================================

test_dataset = FakeNewsDataset(
    test_df,
    tokenizer,
    MAX_LENGTH
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Generate predictions
# ============================================================

print("\nGenerating predictions...")

all_predictions = []

with torch.no_grad():

    for batch in test_loader:

        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }

        outputs = model(**inputs)

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


test_predictions = np.array(all_predictions)

test_results = test_df.copy()
test_results["prediction"] = test_predictions


# ============================================================
# Per-language evaluation
# ============================================================

print("\n" + "=" * 60)
print("PER-LANGUAGE MURIL TEST RESULTS")
print("=" * 60)

for language in sorted(test_results["language"].unique()):

    lang_df = test_results[
        test_results["language"] == language
    ]

    accuracy = accuracy_score(
        lang_df["label"],
        lang_df["prediction"]
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            lang_df["label"],
            lang_df["prediction"],
            average="binary",
            zero_division=0
        )
    )

    print(f"\n{language.upper()}")
    print(f"Samples:   {len(lang_df)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


# ============================================================
# Overall verification
# ============================================================

accuracy = accuracy_score(
    test_results["label"],
    test_results["prediction"]
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        test_results["label"],
        test_results["prediction"],
        average="binary",
        zero_division=0
    )
)

print("\n" + "=" * 60)
print("OVERALL TEST RESULTS")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")