import os
import re
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "faketype_en"
RESULTS_DIR = PROJECT_ROOT / "results"
START_MODEL_DIR = PROJECT_ROOT / "models" / "faketype_en"
SAVE_MODEL_DIR = PROJECT_ROOT / "models" / "faketype_en_continued"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

ID2LABEL = {0: "plain", 1: "satire", 2: "clickbait"}
LABEL2ID = {"plain": 0, "satire": 1, "clickbait": 2}
CLASS_NAMES = ["plain", "satire", "clickbait"]


class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=48):
        self.texts = list(texts)
        self.labels = [LABEL2ID[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2], zero_division=0
    )
    return {
        "acc": acc,
        "macro_f1": f1,
        "macro_p": p,
        "macro_r": r,
        "per_class": {
            ID2LABEL[i]: {"p": per_class_p[i], "r": per_class_r[i], "f1": per_class_f1[i]}
            for i in range(3)
        },
        "preds": all_preds,
        "labels": all_labels
    }


def train_continued(batch_size=32, grad_accum=1, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60, flush=True)
    print("CONTINUED TRAINING: ROBERTA-BASE (EPOCH 2)", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {device} | Batch Size: {batch_size} | Grad Accum: {grad_accum} | LR: {lr}", flush=True)
    print(f"Starting model path (READ-ONLY): {START_MODEL_DIR}", flush=True)
    print(f"Output model path:              {SAVE_MODEL_DIR}", flush=True)

    # 1. Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Loaded splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}", flush=True)

    # 2. Load model & tokenizer locally from saved Epoch 1
    print(f"Loading base checkpoint from {START_MODEL_DIR}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(START_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(START_MODEL_DIR), num_labels=3)
    model.to(device)
    print("Epoch 1 model and tokenizer loaded successfully.", flush=True)

    # Pre-training validation check
    val_dataset = HeadlineDataset(val_df["text"], val_df["label"], tokenizer, max_len=48)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    initial_val = evaluate(model, val_loader, device)
    print(f"Baseline Epoch 1 Validation: Acc={initial_val['acc']:.4f}, Macro-F1={initial_val['macro_f1']:.4f}", flush=True)

    train_dataset = HeadlineDataset(train_df["text"], train_df["label"], tokenizer, max_len=48)
    test_dataset = HeadlineDataset(test_df["text"], test_df["label"], tokenizer, max_len=48)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Fresh Optimizer & Scheduler for continued fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    epochs = 1
    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4. Train 1 additional epoch
    model.train()
    total_loss = 0.0
    step_count = 0
    t0 = time.time()
    print(f"\n--- Starting Additional Epoch (Epoch 2/2) [{len(train_loader)} steps] ---", flush=True)

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum

        scaler.scale(loss).backward()

        if step % grad_accum == 0 or step == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * grad_accum
        step_count += 1

        if step % 100 == 0 or step == len(train_loader):
            avg_l = total_loss / step_count
            lr_cur = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else lr
            print(f"Step {step}/{len(train_loader)} | Loss: {avg_l:.4f} | LR: {lr_cur:.2e} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    # 5. Validation after Epoch 2
    print("\nRunning Validation Evaluation...", flush=True)
    val_res = evaluate(model, val_loader, device)
    print(f"Continued Validation: Acc={val_res['acc']:.4f}, Macro-F1={val_res['macro_f1']:.4f}", flush=True)
    for cls, m in val_res["per_class"].items():
        print(f"  {cls}: P={m['p']:.4f}, R={m['r']:.4f}, F1={m['f1']:.4f}", flush=True)

    # Save continued model separately
    print(f"\nSaving continued model to {SAVE_MODEL_DIR}...", flush=True)
    model.save_pretrained(str(SAVE_MODEL_DIR))
    tokenizer.save_pretrained(str(SAVE_MODEL_DIR))
    print("Model saved successfully.", flush=True)

    # 6. Test evaluation
    print("\nRunning Test Evaluation...", flush=True)
    test_res = evaluate(model, test_loader, device)
    print(f"Final Test Results: Acc={test_res['acc']:.4f}, Macro-F1={test_res['macro_f1']:.4f}", flush=True)
    for cls, m in test_res["per_class"].items():
        print(f"  {cls}: P={m['p']:.4f}, R={m['r']:.4f}, F1={m['f1']:.4f}", flush=True)

    # 7. Confusion matrix CSV
    cm = confusion_matrix(test_res["labels"], test_res["preds"], labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES])
    cm_path = RESULTS_DIR / "faketype_en_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}:\n{cm_df}", flush=True)

    # 8. Val accuracy for headlines with vs without a digit, per class
    print("\nComputing Val accuracy with vs without a digit per class...", flush=True)
    val_preds = val_res["preds"]
    val_df_copy = val_df.copy()
    val_df_copy["pred"] = [ID2LABEL[p] for p in val_preds]
    val_df_copy["has_digit"] = val_df_copy["text"].apply(lambda s: bool(re.search(r"\d", s)))

    digit_val_stats = {}
    for cls in CLASS_NAMES:
        cls_sub = val_df_copy[val_df_copy["label"] == cls]

        sub_digit = cls_sub[cls_sub["has_digit"]]
        acc_digit = (sub_digit["pred"] == cls).mean() if len(sub_digit) > 0 else 0.0

        sub_no_digit = cls_sub[~cls_sub["has_digit"]]
        acc_no_digit = (sub_no_digit["pred"] == cls).mean() if len(sub_no_digit) > 0 else 0.0

        digit_val_stats[cls] = {
            "with_digit_acc": acc_digit,
            "with_digit_n": len(sub_digit),
            "without_digit_acc": acc_no_digit,
            "without_digit_n": len(sub_no_digit),
        }
        print(f"  {cls} | With digit: n={len(sub_digit)}, acc={acc_digit:.4f} | Without digit: n={len(sub_no_digit)}, acc={acc_no_digit:.4f}", flush=True)

    # 9. Write results/faketype_en_metrics.txt
    metrics_path = RESULTS_DIR / "faketype_en_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ROBERTA-BASE FAKETYPE ENGLISH (CONTINUED TRAINING) TEST METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy:   {test_res['acc']:.4f}\n")
        f.write(f"Macro Precision: {test_res['macro_p']:.4f}\n")
        f.write(f"Macro Recall:    {test_res['macro_r']:.4f}\n")
        f.write(f"Macro F1:        {test_res['macro_f1']:.4f}\n\n")
        f.write("Per-Class Test Metrics:\n")
        for cls, m in test_res["per_class"].items():
            f.write(f"  {cls:<12}: P={m['p']:.4f}, R={m['r']:.4f}, F1={m['f1']:.4f}\n")

        f.write("\nValidation Accuracy by Digit Presence per Class:\n")
        f.write("Class        With Digit (Acc / N)     Without Digit (Acc / N)\n")
        f.write("-" * 60 + "\n")
        for cls in CLASS_NAMES:
            st = digit_val_stats[cls]
            w_str = f"{st['with_digit_acc']:.4f} (n={st['with_digit_n']})"
            wo_str = f"{st['without_digit_acc']:.4f} (n={st['without_digit_n']})"
            f.write(f"{cls:<12} {w_str:<24} {wo_str:<24}\n")

    print(f"\nMetrics written successfully to {metrics_path}", flush=True)


def main():
    try:
        train_continued(batch_size=32, grad_accum=1, lr=1e-5)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM Encountered]: {e}. Falling back to batch 16 + gradient accumulation 2...", flush=True)
        torch.cuda.empty_cache()
        train_continued(batch_size=16, grad_accum=2, lr=1e-5)


if __name__ == "__main__":
    main()
