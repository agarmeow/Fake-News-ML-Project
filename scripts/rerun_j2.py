from dotenv import load_dotenv
load_dotenv()

import os
import re
import sys
import time
import json
from pathlib import Path
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ENV RULE: load_dotenv() first; print "GROQ_API_KEY set: True/False"; abort Groq steps loudly if False
api_key = os.environ.get("GROQ_API_KEY")
key_is_set = bool(api_key and api_key.strip())
print(f"GROQ_API_KEY set: {key_is_set}")
if not key_is_set:
    raise RuntimeError("GROQ_API_KEY is not set in environment! Aborting Groq steps.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reasoning.faketype_llm import classify_fake_type, normalize_label

EVAL_CSV_PATH = PROJECT_ROOT / "data" / "eval" / "faketype_test.csv"
TEST_SPLIT_PATH = PROJECT_ROOT / "results" / "faketype_test_split.csv"

ID2LABEL = {0: "ordinary", 1: "satire", 2: "misleading", 3: "clickbait"}
CLASS_NAMES = ["ordinary", "satire", "misleading", "clickbait"]


def evaluate_dataset(texts, raw_golds, dataset_name, sleep_between=0.6):
    print("\n" + "=" * 70)
    print(f"EVALUATING GROQ ON: {dataset_name} (N={len(texts)})")
    print("=" * 70)

    # Normalize gold labels: e.g. "ordinary news" -> "ordinary"
    golds = [normalize_label(g) or str(g).strip().lower() for g in raw_golds]

    preds = []
    t0 = time.time()
    for i, t in enumerate(texts):
        res = classify_fake_type(t)
        p = res.get("label", "unclear")
        preds.append(p)
        time.sleep(sleep_between)

    elapsed = time.time() - t0
    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)
    print(f"Finished {len(texts)} calls in {elapsed:.1f}s: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}")

    print("\nPrediction Counts:")
    for k, v in Counter(preds).most_common():
        print(f"  {k}: {v}")

    # Confusion matrix
    cols = sorted(list(set(preds) | set(CLASS_NAMES)))
    cm_df = pd.DataFrame(index=CLASS_NAMES, columns=cols).fillna(0)
    for g, p in zip(golds, preds):
        if g in cm_df.index and p in cm_df.columns:
            cm_df.loc[g, p] += 1
    print("\nConfusion Matrix (Gold rows x Predicted cols):")
    print(cm_df.to_string())

    return {
        "acc": acc,
        "f1": macro_f1,
        "preds": preds,
        "cm_df": cm_df,
    }


def main():
    # 1. faketype_test.csv (60 rows)
    eval_df = pd.read_csv(EVAL_CSV_PATH)
    eval_texts = eval_df["text"].tolist()
    eval_golds = eval_df["label"].tolist()

    res_eval = evaluate_dataset(eval_texts, eval_golds, "faketype_test.csv (60 rows)", sleep_between=0.5)

    # 2. Fakeddit 200-row sample
    test_split_df = pd.read_csv(TEST_SPLIT_PATH)
    test_split_df["true_label"] = test_split_df["label"].map(ID2LABEL)
    sampled_dfs = []
    for lbl, group in test_split_df.groupby("label"):
        sampled_dfs.append(group.sample(n=min(len(group), 50), random_state=42))
    fakeddit_sample_df = pd.concat(sampled_dfs, ignore_index=True)

    fakeddit_texts = fakeddit_sample_df["clean_title"].tolist()
    fakeddit_golds = fakeddit_sample_df["true_label"].tolist()

    res_fakeddit = evaluate_dataset(fakeddit_texts, fakeddit_golds, "Fakeddit 200-row sample", sleep_between=0.6)


if __name__ == "__main__":
    main()
