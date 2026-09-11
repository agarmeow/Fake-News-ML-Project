import pandas as pd
import numpy as np
import torch

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from torch.utils.data import Dataset


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "google/muril-base-cased"

DATA_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models/muril_classifier")

MAX_LENGTH = 256
NUM_LABELS = 2
RANDOM_SEED = 42


# ============================================================
# Dataset class
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
# Load datasets
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")


# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading MuRIL tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully.")


# ============================================================
# Create PyTorch datasets
# ============================================================

print("\nPreparing datasets...")

train_dataset = FakeNewsDataset(
    train_df,
    tokenizer,
    MAX_LENGTH
)

val_dataset = FakeNewsDataset(
    val_df,
    tokenizer,
    MAX_LENGTH
)

test_dataset = FakeNewsDataset(
    test_df,
    tokenizer,
    MAX_LENGTH
)

print("Datasets prepared.")


# ============================================================
# Load MuRIL model
# ============================================================

print("\nLoading MuRIL model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

print("MuRIL model loaded successfully.")


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# Training configuration
# ============================================================

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    save_total_limit=1,

    report_to="none",

    seed=RANDOM_SEED
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics
)


# ============================================================
# Train
# ============================================================

print("\n" + "=" * 60)
print("STARTING MURIL TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# Validation results
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

val_results = trainer.evaluate(
    eval_dataset=val_dataset
)

for key, value in val_results.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# Test results
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

test_results = trainer.evaluate(
    eval_dataset=test_dataset
)

for key, value in test_results.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# Save final model
# ============================================================

print("\nSaving MuRIL model...")

trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

print(f"Model saved to: {MODEL_DIR}")

# ============================================================
# Per-language test evaluation
# ============================================================

print("\n" + "=" * 60)
print("PER-LANGUAGE MURIL TEST RESULTS")
print("=" * 60)

predictions_output = trainer.predict(test_dataset)

test_predictions = np.argmax(
    predictions_output.predictions,
    axis=1
)

test_results_df = test_df.copy()
test_results_df["prediction"] = test_predictions

for language in sorted(test_results_df["language"].unique()):

    lang_df = test_results_df[
        test_results_df["language"] == language
    ]

    accuracy = accuracy_score(
        lang_df["label"],
        lang_df["prediction"]
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        lang_df["label"],
        lang_df["prediction"],
        average="binary",
        zero_division=0
    )

    print(f"\n{language.upper()}")
    print(f"Samples:   {len(lang_df)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")