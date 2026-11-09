from dotenv import load_dotenv
load_dotenv()

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "multimodal_train.tsv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models" / "faketype_muril"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 64
RANDOM_SEED = 42

LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}
ID2LABEL = {0: "ordinary", 1: "satire", 2: "misleading", 3: "clickbait"}
LABEL2ID = {"ordinary": 0, "satire": 1, "misleading": 2, "clickbait": 3}


class FakeTypeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        print(f"Pre-tokenizing {len(texts)} samples (max_length={max_length})...")
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_and_preprocess_data():
    print(f"Reading {DATA_PATH}...")
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        usecols=["clean_title", "6_way_label"]
    )
    # Keep classes 0, 1, 2, 4
    df = df[df["6_way_label"].isin(LABEL_MAP.keys())].copy()
    df["label"] = df["6_way_label"].map(LABEL_MAP)
    df["clean_title"] = df["clean_title"].fillna("").astype(str)

    # Drop empty/very short titles (<4 words) and duplicate titles BEFORE splitting
    df = df[df["clean_title"].str.strip().str.split().str.len() >= 4]
    df = df.drop_duplicates(subset=["clean_title"])

    # Sample up to 5000 per class, seed 42
    sampled_dfs = []
    for lbl, group in df.groupby("label"):
        sampled_dfs.append(group.sample(n=min(len(group), 5000), random_state=RANDOM_SEED))
    df = pd.concat(sampled_dfs, ignore_index=True)

    # Stratified 80/10/10 split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"]
    )

    test_df = test_df.reset_index(drop=True)
    test_df.to_csv(RESULTS_DIR / "faketype_test_split.csv", index=False)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df
    )


def run_audit(train_df, val_df, test_df):
    print("\nRunning TF-IDF + LogisticRegression audit...")
    audit_file = RESULTS_DIR / "faketype_audit.txt"

    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train = vec.fit_transform(train_df["clean_title"])
    y_train = train_df["label"].values

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    feature_names = np.array(vec.get_feature_names_out())

    lines = []
    lines.append("=" * 60)
    lines.append("FAKETYPE DATASET AUDIT & TOP FEATURES PER CLASS")
    lines.append("=" * 60)
    lines.append("\nSPLIT COUNTS:")
    for split_name, split_df in [("Train (80%)", train_df), ("Val (10%)", val_df), ("Test (10%)", test_df)]:
        counts = split_df["label"].value_counts().sort_index()
        counts_str = ", ".join(f"{ID2LABEL[k]}: {v}" for k, v in counts.items())
        lines.append(f"  {split_name:<12}: Total={len(split_df)} | {counts_str}")

    lines.append("\nTOP 20 TF-IDF+LR FEATURES PER CLASS:")
    for class_idx in range(4):
        class_name = ID2LABEL[class_idx]
        coefs = clf.coef_[class_idx]
        top_indices = np.argsort(coefs)[::-1][:20]
        top_words = feature_names[top_indices]
        lines.append(f"\nClass {class_idx} ({class_name}):")
        lines.append(", ".join(top_words))

    output_text = "\n".join(lines)
    with open(audit_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(output_text)
    print(f"\nAudit saved to {audit_file}")
    return audit_file


def run_training(train_df, val_df, test_df):
    print("\nStarting MuRIL training (G2)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    train_dataset = FakeTypeDataset(train_df["clean_title"], train_df["label"], tokenizer, MAX_LENGTH)
    val_dataset = FakeTypeDataset(val_df["clean_title"], val_df["label"], tokenizer, MAX_LENGTH)
    test_dataset = FakeTypeDataset(test_df["clean_title"], test_df["label"], tokenizer, MAX_LENGTH)

    # Class-weighted loss
    class_weights_vals = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2, 3]),
        y=train_df["label"].values
    )
    class_weights = torch.tensor(class_weights_vals, dtype=torch.float)
    print(f"Class weights: {class_weights_vals}")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        dataloader_pin_memory=True,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=RANDOM_SEED
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    trainer.train()

    # Save final best model and tokenizer
    print(f"Saving fine-tuned model to {MODEL_DIR}...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Test evaluation
    print("\nEvaluating on test set...")
    test_preds_raw = trainer.predict(test_dataset)
    test_preds = np.argmax(test_preds_raw.predictions, axis=1)
    test_labels = test_df["label"].values

    acc = accuracy_score(test_labels, test_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average=None, zero_division=0
    )

    cm = confusion_matrix(test_labels, test_preds)

    # Save metrics text
    metrics_path = RESULTS_DIR / "faketype_muril_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("MURIL FAKETYPE CLASSIFIER TEST METRICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {p_macro:.4f}\n")
        f.write(f"Macro Recall: {r_macro:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n\n")
        f.write("Per-class Metrics:\n")
        for i in range(4):
            f.write(f"  {ID2LABEL[i]:<12}: P={per_class_p[i]:.4f}, R={per_class_r[i]:.4f}, F1={per_class_f1[i]:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(test_labels, test_preds, target_names=[ID2LABEL[i] for i in range(4)]))

    # Save Confusion Matrix CSV
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{ID2LABEL[i]}" for i in range(4)],
        columns=[f"pred_{ID2LABEL[i]}" for i in range(4)]
    )
    cm_csv_path = RESULTS_DIR / "faketype_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    # Save Confusion Matrix PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues")
        plt.title("MuRIL Fake-Type Confusion Matrix", pad=20)
        fig.colorbar(cax)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([ID2LABEL[i] for i in range(4)], rotation=45, ha="left")
        ax.set_yticklabels([ID2LABEL[i] for i in range(4)])
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_png_path = RESULTS_DIR / "faketype_confusion_matrix.png"
        plt.savefig(cm_png_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: could not generate confusion matrix PNG: {e}")

    print("\nTest Evaluation Complete:")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1_macro:.4f}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true", help="Run only data audit")
    args = parser.parse_args()

    train_df, val_df, test_df = load_and_preprocess_data()
    run_audit(train_df, val_df, test_df)

    if not args.audit_only:
        run_training(train_df, val_df, test_df)
