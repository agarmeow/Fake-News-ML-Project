from dotenv import load_dotenv
load_dotenv()

import os
import re
import sys
import time
import json
from pathlib import Path
import pandas as pd
import requests
import urllib3
from collections import Counter

# Set stdout encoding to utf-8 with replacement
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ENV RULE: load_dotenv() first; print "GROQ_API_KEY set: True/False"; abort Groq steps loudly if False
api_key = os.environ.get("GROQ_API_KEY")
key_is_set = bool(api_key and api_key.strip())
print(f"GROQ_API_KEY set: {key_is_set}")
if not key_is_set:
    raise RuntimeError("GROQ_API_KEY is not set in environment! Aborting Groq steps.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reasoning.faketype_llm import (
    classify_fake_type,
    CHAT_URL,
    SYSTEM_PROMPT,
    VALID_LABELS,
    _lenient_json_parse,
)

EVAL_CSV_PATH = PROJECT_ROOT / "data" / "eval" / "faketype_test.csv"
TEST_SPLIT_PATH = PROJECT_ROOT / "results" / "faketype_test_split.csv"

ID2LABEL = {0: "ordinary", 1: "satire", 2: "misleading", 3: "clickbait"}


def inspect_groq_run(texts, gold_labels, run_name, sleep_time=0.1):
    print("=" * 70)
    print(f"RUN: {run_name} (N={len(texts)})")
    print("=" * 70)

    model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    temperature = 0.1  # current temperature in faketype_llm.py

    print(f"(d) Groq model name: {model_name}")
    print(f"    Temperature: {temperature}")
    print(f"    Sleep between calls: {sleep_time}s")

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    http_status_counts = Counter()
    raw_responses = []
    predictions = []
    rationales = []

    for i, text in enumerate(texts):
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text to classify:\n{text}"},
            ],
            "temperature": temperature,
            "max_completion_tokens": 1024,
            "response_format": {"type": "json_object"},
        }

        status = "unknown"
        raw_content = ""
        try:
            resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=10, verify=False)
            status = f"HTTP_{resp.status_code}"
            http_status_counts[status] += 1
            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"].strip()
            else:
                raw_content = f"ERROR: {resp.status_code} - {resp.text[:200]}"
        except requests.exceptions.Timeout:
            http_status_counts["TIMEOUT"] += 1
            status = "TIMEOUT"
            raw_content = "ERROR: Request timed out"
        except Exception as e:
            err_name = type(e).__name__
            http_status_counts[f"ERR_{err_name}"] += 1
            status = f"ERR_{err_name}"
            raw_content = f"ERROR: {e}"

        raw_responses.append(raw_content)

        # Parse with project parser
        parsed = _lenient_json_parse(raw_content)
        pred_label = "unclear"
        rat = ""
        if parsed and isinstance(parsed, dict):
            lbl = str(parsed.get("label", "")).strip().lower()
            if lbl in VALID_LABELS:
                pred_label = lbl
                rat = str(parsed.get("rationale", "")).strip()
            else:
                pred_label = f"invalid({lbl})"
        predictions.append(pred_label)
        rationales.append(rat)

        time.sleep(sleep_time)

    # (a) prediction counts per label incl. "unclear", plus error/timeout/HTTP status counts
    print(f"\n(a) Prediction counts per label:")
    pred_counter = Counter(predictions)
    for lbl, cnt in pred_counter.most_common():
        print(f"    {lbl}: {cnt}")

    print(f"\n    HTTP status / Error counts:")
    for st, cnt in http_status_counts.most_common():
        print(f"    {st}: {cnt}")

    # (b) 5 raw model responses (first 300 chars) from 5 different rows
    print(f"\n(b) 5 raw model responses (first 300 chars):")
    sample_indices = [0, len(texts)//4, len(texts)//2, 3*len(texts)//4, len(texts)-1]
    for idx in sample_indices:
        snippet = raw_responses[idx][:300].replace("\n", " ")
        print(f"    Row {idx} (Gold: {gold_labels[idx]}): {snippet}")

    # (c) set of gold labels and predicted labels after normalization, and mismatch
    raw_gold_set = set(gold_labels)
    pred_set = set(predictions)
    print(f"\n(c) Raw gold labels set: {sorted(list(raw_gold_set))}")
    print(f"    Predicted labels set: {sorted(list(pred_set))}")
    unmatchable = [g for g in raw_gold_set if g not in VALID_LABELS]
    print(f"    Unmatchable gold labels (not in VALID_LABELS): {unmatchable}")

    # (e) full confusion matrix (gold rows x predicted cols)
    print(f"\n(e) Confusion Matrix (Gold rows x Predicted cols):")
    all_col_labels = sorted(list(set(predictions) | VALID_LABELS))
    # Map gold if necessary or show raw
    unique_golds = sorted(list(raw_gold_set))
    cm_df = pd.DataFrame(index=unique_golds, columns=all_col_labels).fillna(0)
    for g, p in zip(gold_labels, predictions):
        cm_df.loc[g, p] += 1
    print(cm_df.to_string())

    return {
        "predictions": predictions,
        "raw_responses": raw_responses,
        "cm_df": cm_df,
    }


def main():
    # 1. faketype_test.csv (60 rows)
    eval_df = pd.read_csv(EVAL_CSV_PATH)
    eval_texts = eval_df["text"].tolist()
    eval_golds = eval_df["label"].tolist()

    res_eval = inspect_groq_run(eval_texts, eval_golds, "faketype_test.csv (60 rows)")

    # 2. 200-row Fakeddit sample
    test_split_df = pd.read_csv(TEST_SPLIT_PATH)
    test_split_df["true_label"] = test_split_df["label"].map(ID2LABEL)
    sampled_dfs = []
    for lbl, group in test_split_df.groupby("label"):
        sampled_dfs.append(group.sample(n=min(len(group), 50), random_state=42))
    fakeddit_sample_df = pd.concat(sampled_dfs, ignore_index=True)

    fakeddit_texts = fakeddit_sample_df["clean_title"].tolist()
    fakeddit_golds = fakeddit_sample_df["true_label"].tolist()

    res_fakeddit = inspect_groq_run(fakeddit_texts, fakeddit_golds, "Fakeddit 200-row sample")


if __name__ == "__main__":
    main()
