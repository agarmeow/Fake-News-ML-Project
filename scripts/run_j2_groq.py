from dotenv import load_dotenv
load_dotenv()

import os
import re
import sys
import time
import json
import csv
from pathlib import Path
import pandas as pd
import requests
import urllib3
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ENV RULE: load_dotenv() first; print "GROQ_API_KEY set: True/False"; abort Groq steps loudly if False
api_key = os.environ.get("GROQ_API_KEY")
key_is_set = bool(api_key and api_key.strip())
print(f"GROQ_API_KEY set: {key_is_set}", flush=True)
if not key_is_set:
    raise RuntimeError("GROQ_API_KEY is not set in environment! Aborting Groq steps.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reasoning.faketype_llm import (
    CHAT_URL,
    SYSTEM_PROMPT,
    VALID_LABELS,
    normalize_label,
    _lenient_json_parse,
)

EVAL_CSV_PATH = PROJECT_ROOT / "data" / "eval" / "faketype_test.csv"
OUT_CSV_PATH = PROJECT_ROOT / "results" / "groq_faketype_60.csv"

CLASS_NAMES = ["ordinary", "satire", "misleading", "clickbait"]


def call_groq_single(text, model_name, headers):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text to classify:\n{text}"},
        ],
        "temperature": 0.0,
        "max_completion_tokens": 150,
        "response_format": {"type": "json_object"},
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=15, verify=False)
            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"].strip()
                parsed = _lenient_json_parse(raw_content)
                if parsed and isinstance(parsed, dict):
                    norm = normalize_label(parsed.get("label"))
                    if norm:
                        rat = str(parsed.get("rationale", "")).strip() or "Groq classification."
                        return {"label": norm, "rationale": rat, "status": "OK"}
                return {"label": "unclear", "rationale": f"Unparsed output: {raw_content[:100]}", "status": "PARSE_ERROR"}
            elif resp.status_code == 429:
                err_text = resp.text
                retry_after_hdr = resp.headers.get("retry-after")
                retry_after_sec = float(retry_after_hdr) if retry_after_hdr else 0.0

                # Check for daily-limit or quota exhaustion messages
                if "daily" in err_text.lower() or "quota" in err_text.lower() or retry_after_sec > 60:
                    msg = f"FATAL 429: Hard rate limit or daily quota reached ({err_text[:200]}). retry-after={retry_after_sec}s."
                    print(msg, flush=True)
                    sys.exit(1)

                backoff = min(20.0, max(2.0, retry_after_sec if retry_after_sec > 0 else (2.0 ** attempt + 2.0)))
                time.sleep(backoff)
            else:
                time.sleep(1.5)
        except requests.exceptions.Timeout:
            time.sleep(min(20.0, 2.0 ** attempt + 2.0))
        except requests.exceptions.RequestException as e:
            time.sleep(2.0)
        except Exception as e:
            time.sleep(1.0)

    return {"label": "unclear", "rationale": "Max retries exhausted.", "status": "ERROR"}


def main():
    if not EVAL_CSV_PATH.exists():
        print(f"Error: {EVAL_CSV_PATH} not found!", flush=True)
        sys.exit(1)

    eval_df = pd.read_csv(EVAL_CSV_PATH)
    total_rows = len(eval_df)
    print(f"Loaded {total_rows} rows from {EVAL_CSV_PATH}.", flush=True)

    # Resume capability: check results/groq_faketype_60.csv
    processed_texts = set()
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV_PATH.exists():
        try:
            existing_df = pd.read_csv(OUT_CSV_PATH)
            if "text" in existing_df.columns:
                processed_texts = set(existing_df["text"].tolist())
                print(f"Found {len(processed_texts)} already processed rows in {OUT_CSV_PATH}. Resuming...", flush=True)
        except Exception:
            pass

    # Initialize CSV header if not exists
    if not OUT_CSV_PATH.exists() or OUT_CSV_PATH.stat().st_size == 0:
        with open(OUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "gold", "pred", "rationale", "language"])

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    print(f"Using Groq model: {model_name} (temperature=0.0)", flush=True)

    consecutive_failures = 0

    for idx, row in eval_df.iterrows():
        text = str(row["text"]).strip()
        raw_gold = str(row["label"]).strip()
        gold_norm = normalize_label(raw_gold) or raw_gold.lower()
        lang = str(row.get("language", "en")).strip()

        if text in processed_texts:
            continue

        res = call_groq_single(text, model_name, headers)
        pred = res["label"]
        rat = res["rationale"]

        if res["status"] in ("OK",) and pred in VALID_LABELS:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print(f"ABORT: 5 consecutive failures reached at row {idx+1}. Aborting.", flush=True)
                sys.exit(2)

        # Append prediction to CSV immediately
        with open(OUT_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([text, gold_norm, pred, rat, lang])

        processed_texts.add(text)

        # Print progress every 10 rows
        count_done = len(processed_texts)
        if count_done % 10 == 0 or count_done == total_rows:
            print(f"[Progress] Processed {count_done}/{total_rows} rows...", flush=True)

        time.sleep(1.2)  # Gentle spacing to stay below 8000 TPM limit

    # Load final results and print confusion matrix & counts
    final_df = pd.read_csv(OUT_CSV_PATH)
    print("\n" + "=" * 50, flush=True)
    print("FINAL EVALUATION RESULTS", flush=True)
    print("=" * 50, flush=True)

    pred_counts = Counter(final_df["pred"])
    print(f"Prediction counts per label: {dict(pred_counts)}", flush=True)

    # Confusion matrix
    golds = final_df["gold"].tolist()
    preds = final_df["pred"].tolist()

    all_cols = CLASS_NAMES + [c for c in sorted(list(set(preds))) if c not in CLASS_NAMES]
    cm_df = pd.DataFrame(0, index=CLASS_NAMES, columns=all_cols)
    for g, p in zip(golds, preds):
        if g in cm_df.index and p in cm_df.columns:
            cm_df.loc[g, p] += 1

    print("\nConfusion Matrix (Gold rows x Predicted cols):", flush=True)
    print(cm_df.to_string(), flush=True)

    valid_mask = [p in CLASS_NAMES for p in preds]
    if any(valid_mask):
        acc = accuracy_score(golds, preds)
        macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)
        print(f"\nOverall Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
