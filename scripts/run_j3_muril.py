import os
import re
import sys
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "faketype_muril"
EVAL_CSV_PATH = PROJECT_ROOT / "data" / "eval" / "faketype_test.csv"
OUT_CSV_PATH = PROJECT_ROOT / "results" / "muril_faketype_60.csv"

ID2LABEL = {0: "ordinary", 1: "satire", 2: "misleading", 3: "clickbait"}
CLASS_NAMES = ["ordinary", "satire", "misleading", "clickbait"]

# 1. Load data
df = pd.read_csv(EVAL_CSV_PATH)

# Map label "ordinary news" -> "ordinary"
df["gold"] = df["label"].astype(str).str.strip().str.lower()
df["gold"] = df["gold"].replace({"ordinary news": "ordinary"})

# Skip rows without Devanagari if language is hi/mr
indic_mask = df["language"].astype(str).str.lower().isin(["hi", "mr"])
has_devanagari = df["text"].astype(str).apply(lambda t: bool(re.search(r"[\u0900-\u097F]", t)))
corrupted_mask = indic_mask & (~has_devanagari)
skipped_count = int(corrupted_mask.sum())

if skipped_count > 0:
    df = df[~corrupted_mask].copy()

df = df.reset_index(drop=True)

# 2. Load MuRIL
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()

# 3. Predict
inputs = tokenizer(
    df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors="pt"
).to(device)

with torch.inference_mode():
    outputs = model(**inputs)
    preds_idx = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    preds = [ID2LABEL[i] for i in preds_idx]

df["pred"] = preds

# 4. Save results/muril_faketype_60.csv (columns: text, gold, pred, language)
out_df = df[["text", "gold", "pred", "language"]].copy()
out_df.to_csv(OUT_CSV_PATH, index=False)

# 5. Metrics
pred_counts = pd.Series(preds).value_counts().to_dict()
overall_acc = accuracy_score(df["gold"], preds)
overall_f1 = f1_score(df["gold"], preds, average="macro", zero_division=0)

en_mask = df["language"].astype(str).str.lower() == "en"
en_acc = accuracy_score(df.loc[en_mask, "gold"], [p for p, m in zip(preds, en_mask) if m])

indic_eval_mask = df["language"].astype(str).str.lower().isin(["hi", "mr"])
indic_acc = accuracy_score(df.loc[indic_eval_mask, "gold"], [p for p, m in zip(preds, indic_eval_mask) if m])

# 4x4 Confusion Matrix
cm = confusion_matrix(df["gold"], preds, labels=CLASS_NAMES)
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

print(f"Skipped Indic rows (no Devanagari): {skipped_count}")
print(f"Predicted counts: {pred_counts}")
print(f"Overall Acc: {overall_acc:.4f} | Overall Macro-F1: {overall_f1:.4f}")
print(f"English Acc (n={en_mask.sum()}): {en_acc:.4f} | Indic Acc (n={indic_eval_mask.sum()}): {indic_acc:.4f}")
print("Confusion Matrix (Gold rows x Pred cols):")
print(cm_df)
