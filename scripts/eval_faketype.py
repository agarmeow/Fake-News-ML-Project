from dotenv import load_dotenv
load_dotenv()

import os
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models" / "faketype_muril"
EVAL_CSV_PATH = PROJECT_ROOT / "data" / "eval" / "faketype_test.csv"
TEST_SPLIT_PATH = RESULTS_DIR / "faketype_test_split.csv"

# Check GROQ_API_KEY at start of G4
api_key = os.environ.get("GROQ_API_KEY")
key_is_set = bool(api_key and api_key.strip())
print(f"GROQ_API_KEY set: {key_is_set}")
if not key_is_set:
    raise RuntimeError("GROQ_API_KEY is not set in environment! Aborting Groq steps.")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from reasoning.faketype_llm import classify_fake_type

from transformers import AutoTokenizer, AutoModelForSequenceClassification

ID2LABEL = {0: "ordinary", 1: "satire", 2: "misleading", 3: "clickbait"}
LABEL2ID = {"ordinary": 0, "satire": 1, "misleading": 2, "clickbait": 3}
VALID_LABELS = {"ordinary", "satire", "misleading", "clickbait"}


def load_muril_model():
    print(f"\nLoading fine-tuned MuRIL from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_muril(texts, tokenizer, model, device, batch_size=32, max_length=64):
    preds = []
    text_list = list(texts)
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            preds.extend([ID2LABEL[idx] for idx in batch_preds])
    return preds


def evaluate_part_a(tokenizer, model, device):
    print("\n" + "=" * 60)
    print("(a) FAKEDDIT TEST SPLIT EVALUATION")
    print("=" * 60)

    if not TEST_SPLIT_PATH.exists():
        print(f"Error: {TEST_SPLIT_PATH} not found!")
        return None, None

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    test_df["true_label"] = test_df["label"].map(ID2LABEL)
    print(f"Loaded {len(test_df)} test split samples.")

    # MuRIL on ALL
    print("Running MuRIL on all test samples...")
    t0 = time.time()
    muril_preds = predict_muril(test_df["clean_title"], tokenizer, model, device)
    muril_time = time.time() - t0
    y_true_all = test_df["true_label"].values
    muril_acc = accuracy_score(y_true_all, muril_preds)
    muril_macro_f1 = f1_score(y_true_all, muril_preds, average="macro", zero_division=0)
    print(f"MuRIL (all {len(test_df)} rows) in {muril_time:.2f}s: Acc={muril_acc:.4f}, Macro-F1={muril_macro_f1:.4f}")

    # Groq on stratified 200-row sample
    print("\nSampling stratified 200-row sample for Groq evaluation...")
    sampled_dfs = []
    for lbl, group in test_df.groupby("label"):
        sampled_dfs.append(group.sample(n=min(len(group), 50), random_state=42))
    sample_df = pd.concat(sampled_dfs, ignore_index=True)

    print(f"Running Groq few-shot on {len(sample_df)} samples...")
    groq_preds = []
    groq_rationales = []
    t0 = time.time()
    for idx, row in sample_df.iterrows():
        title = row["clean_title"]
        try:
            res = classify_fake_type(title)
            groq_preds.append(res.get("label", "unclear"))
            groq_rationales.append(res.get("rationale", ""))
        except Exception as e:
            groq_preds.append("unclear")
            groq_rationales.append(f"Error: {e}")
        time.sleep(0.1)  # Respect rate limits

    groq_time = time.time() - t0
    y_true_sample = sample_df["true_label"].values
    groq_acc = accuracy_score(y_true_sample, groq_preds)
    groq_macro_f1 = f1_score(y_true_sample, groq_preds, average="macro", zero_division=0)
    print(f"Groq (200-row sample) in {groq_time:.2f}s: Acc={groq_acc:.4f}, Macro-F1={groq_macro_f1:.4f}")

    return {
        "muril_acc": muril_acc,
        "muril_macro_f1": muril_macro_f1,
        "groq_acc": groq_acc,
        "groq_macro_f1": groq_macro_f1,
        "sample_df": sample_df,
        "groq_preds": groq_preds,
    }


def evaluate_part_b(tokenizer, model, device):
    print("\n" + "=" * 60)
    print("(b) MULTILINGUAL EVALUATION (data/eval/faketype_test.csv)")
    print("=" * 60)

    if not EVAL_CSV_PATH.exists():
        print(f"File {EVAL_CSV_PATH} is missing, skipping (b).")
        return None

    df = pd.read_csv(EVAL_CSV_PATH)
    print(f"Loaded {len(df)} rows from {EVAL_CSV_PATH}.")

    # Label hygiene: strip and lowercase labels; map "ordinary news" -> "ordinary"
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].replace({"ordinary news": "ordinary"})

    # Check for corrupted Indic rows (hi/mr without Devanagari U+0900-U+097F)
    indic_mask = df["language"].astype(str).str.lower().isin(["hi", "mr"])
    has_devanagari = df["text"].astype(str).apply(lambda t: bool(re.search(r"[\u0900-\u097F]", t)))

    corrupted_mask = indic_mask & (~has_devanagari)
    corrupted_count = int(corrupted_mask.sum())

    if corrupted_count > 0:
        print(f"\n[WARNING] {corrupted_count} Indic rows skipped (corrupted)")
        df = df[~corrupted_mask].copy()

    df = df.reset_index(drop=True)
    print(f"Cleaned evaluation dataset size: {len(df)} rows.")
    print("Class distribution:\n", df["label"].value_counts())
    print("Language distribution:\n", df["language"].value_counts())

    # Run MuRIL
    print("\nRunning MuRIL predictions on eval set...")
    muril_preds = predict_muril(df["text"], tokenizer, model, device)
    df["muril_pred"] = muril_preds

    # Run Groq
    print("Running Groq predictions on eval set...")
    groq_preds = []
    groq_rationales = []
    for idx, row in df.iterrows():
        text = row["text"]
        try:
            res = classify_fake_type(text)
            groq_preds.append(res.get("label", "unclear"))
            groq_rationales.append(res.get("rationale", ""))
        except Exception as e:
            groq_preds.append("unclear")
            groq_rationales.append(f"Error: {e}")
        time.sleep(0.1)

    df["groq_pred"] = groq_preds
    df["groq_rationale"] = groq_rationales

    # Save to results/faketype_eval.csv
    eval_csv_out = RESULTS_DIR / "faketype_eval.csv"
    df.to_csv(eval_csv_out, index=False)
    print(f"Detailed predictions saved to {eval_csv_out}")

    # Compute overall and per-language metrics
    metrics_records = []
    y_true_all = df["label"].values
    metrics_records.append({
        "Language": "Overall (all)",
        "Count": len(df),
        "MuRIL Acc": f"{accuracy_score(y_true_all, df['muril_pred']):.4f}",
        "MuRIL Macro-F1": f"{f1_score(y_true_all, df['muril_pred'], average='macro', zero_division=0):.4f}",
        "Groq Acc": f"{accuracy_score(y_true_all, df['groq_pred']):.4f}",
        "Groq Macro-F1": f"{f1_score(y_true_all, df['groq_pred'], average='macro', zero_division=0):.4f}",
    })

    for lang, group in df.groupby("language"):
        y_true_lang = group["label"].values
        metrics_records.append({
            "Language": lang,
            "Count": len(group),
            "MuRIL Acc": f"{accuracy_score(y_true_lang, group['muril_pred']):.4f}",
            "MuRIL Macro-F1": f"{f1_score(y_true_lang, group['muril_pred'], average='macro', zero_division=0):.4f}",
            "Groq Acc": f"{accuracy_score(y_true_lang, group['groq_pred']):.4f}",
            "Groq Macro-F1": f"{f1_score(y_true_lang, group['groq_pred'], average='macro', zero_division=0):.4f}",
        })

    metrics_df = pd.DataFrame(metrics_records)
    print("\n" + "=" * 60)
    print("MULTILINGUAL EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    return metrics_df


if __name__ == "__main__":
    tokenizer, model, device = load_muril_model()
    part_a_res = evaluate_part_a(tokenizer, model, device)
    part_b_res = evaluate_part_b(tokenizer, model, device)
