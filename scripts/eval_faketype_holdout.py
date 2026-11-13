import os
import sys
import re
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure stdout and stderr handle utf-8 safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY set: {bool(groq_key)}")
if not groq_key:
    print("FATAL: GROQ_API_KEY is not set in environment! Aborting Groq steps loudly.")
    sys.exit(1)

# Configure huggingface_hub with hard 60s timeout and SSL workaround
import httpx
from huggingface_hub.utils import _http
_http.set_client_factory(lambda: httpx.Client(verify=False, follow_redirects=True, timeout=60.0))

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reasoning import faketype_llm
from reasoning.faketype_pipeline import classify_type, classify_type_v2
from fact_check.fact_checker import fact_check

CLASSES = ["ordinary", "satire", "clickbait", "misleading"]

# Claim cache for fact_check
FC_CACHE = {}

def cached_fact_check(claim: str):
    c = claim.strip()
    if c not in FC_CACHE:
        try:
            FC_CACHE[c] = fact_check(c)
        except Exception as e:
            print(f"Warning in fact_check for '{c[:40]}...': {e}")
            FC_CACHE[c] = {"verdict": "UNVERIFIABLE", "error": str(e)}
    return FC_CACHE[c]

# Cache for LLM classification to ensure consistency and avoid redundant Groq calls
_orig_classify = faketype_llm.classify
_LLM_CACHE = {}

def cached_classify(text, model=None):
    t = text.strip()
    if t not in _LLM_CACHE:
        try:
            _LLM_CACHE[t] = _orig_classify(text, model=model)
        except SystemExit:
            # If primary model hits TPD limit and exits, fall back to secondary model
            print(f"Warning: Primary model hit limit on '{t[:40]}...'. Falling back to qwen/qwen3.8-27b.")
            faketype_llm.CONSECUTIVE_FAILURES = 0
            _LLM_CACHE[t] = _orig_classify(text, model="qwen/qwen3.8-27b")
    return _LLM_CACHE[t]

faketype_llm.classify = cached_classify
import reasoning.faketype_pipeline as fpipe
fpipe.classify = cached_classify


def clean_label(lbl: str) -> str:
    if not isinstance(lbl, str):
        return "ordinary"
    s = lbl.strip().lower()
    if s in ("ordinary news", "plain", "factual", "normal"):
        return "ordinary"
    if s not in CLASSES:
        return "ordinary"
    return s


def compute_metrics(golds, preds):
    acc = accuracy_score(golds, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        golds, preds, labels=CLASSES, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        golds, preds, labels=CLASSES, average=None, zero_division=0
    )
    cm = confusion_matrix(golds, preds, labels=CLASSES)
    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "macro_p": p_macro,
        "macro_r": r_macro,
        "per_class": {
            c: {"p": p_cls[i], "r": r_cls[i], "f1": f1_cls[i]}
            for i, c in enumerate(CLASSES)
        },
        "cm": cm,
    }


def format_cm(cm):
    header = f"{'Gold \\ Pred':<14} | " + " | ".join(f"{c:<10}" for c in CLASSES)
    sep = "-" * len(header)
    lines = [header, sep]
    for i, c in enumerate(CLASSES):
        row_str = f"{c:<14} | " + " | ".join(f"{cm[i][j]:<10}" for j in range(len(CLASSES)))
        lines.append(row_str)
    return "\n".join(lines)


def main():
    holdout_path = PROJECT_ROOT / "data" / "eval" / "faketype_holdout.csv"
    if not holdout_path.exists():
        print(f"FATAL: {holdout_path} does not exist!")
        sys.exit(1)
    print(f"Confirmed: {holdout_path} exists.")

    df = pd.read_csv(holdout_path)
    en_df = df[df["language"] == "en"].copy().reset_index(drop=True)
    n_rows = len(en_df)
    print(f"Running evaluation on {n_rows} English holdout rows.")

    out_csv = PROJECT_ROOT / "results" / "faketype_en_holdout.csv"

    # Resume from existing if present
    records = []
    done_texts = set()
    if out_csv.exists():
        try:
            prev_df = pd.read_csv(out_csv)
            if len(prev_df) > 0 and "v2_pred" in prev_df.columns:
                records = prev_df.to_dict("records")
                done_texts = set(r["text"] for r in records)
                print(f"Found {len(records)} existing predictions in {out_csv.name}.")
        except Exception:
            records = []

    for idx, row in en_df.iterrows():
        text = str(row["text"])
        gold = clean_label(row["label"])

        if text in done_texts:
            continue

        print(f"[{idx+1}/{n_rows}] Eval ({gold}): {text[:50]}...")

        # 1. LLM-only
        llm_res = cached_classify(text)
        llm_pred = clean_label(llm_res.get("label", "ordinary"))
        llm_conf = float(llm_res.get("confidence", 0.0))

        # 2. v1 Pipeline
        v1_res = classify_type(text, verifier_fn=cached_fact_check)
        v1_pred = clean_label(v1_res.get("label", "ordinary"))
        v1_source = v1_res.get("source", "default")
        v1_verdict = v1_res.get("verdict")

        # 3. v2 Pipeline
        v2_res = classify_type_v2(text, verifier_fn=cached_fact_check)
        v2_pred = clean_label(v2_res.get("label", "ordinary"))
        v2_source = v2_res.get("source", "default")
        v2_verdict = v2_res.get("verdict")

        rec = {
            "text": text,
            "gold": gold,
            "llm_pred": llm_pred,
            "llm_conf": llm_conf,
            "v1_pred": v1_pred,
            "v1_source": v1_source,
            "v1_verdict": str(v1_verdict),
            "v2_pred": v2_pred,
            "v2_source": v2_source,
            "v2_verdict": str(v2_verdict),
        }
        records.append(rec)
        pd.DataFrame(records).to_csv(out_csv, index=False)

    print(f"\nAll {len(records)} holdout evaluations completed.")

    res_df = pd.DataFrame(records)
    golds = [clean_label(g) for g in res_df["gold"]]
    v1_preds = [clean_label(p) for p in res_df["v1_pred"]]
    v2_preds = [clean_label(p) for p in res_df["v2_pred"]]
    llm_preds = [clean_label(p) for p in res_df["llm_pred"]]

    m_v1 = compute_metrics(golds, v1_preds)
    m_v2 = compute_metrics(golds, v2_preds)
    m_llm = compute_metrics(golds, llm_preds)

    print("\n" + "=" * 60)
    print("HOLDOUT EVALUATION SUMMARY (N=24)")
    print("=" * 60)

    # 1. Overall comparison table
    print(f"\n{'Model':<12} | {'Accuracy':<10} | {'Macro F1':<10} | {'Macro Prec':<10} | {'Macro Rec':<10}")
    print("-" * 62)
    print(f"{'LLM-only':<12} | {m_llm['accuracy']:<10.4f} | {m_llm['macro_f1']:<10.4f} | {m_llm['macro_p']:<10.4f} | {m_llm['macro_r']:<10.4f}")
    print(f"{'v1 Pipeline':<12} | {m_v1['accuracy']:<10.4f} | {m_v1['macro_f1']:<10.4f} | {m_v1['macro_p']:<10.4f} | {m_v1['macro_r']:<10.4f}")
    print(f"{'v2 Pipeline':<12} | {m_v2['accuracy']:<10.4f} | {m_v2['macro_f1']:<10.4f} | {m_v2['macro_p']:<10.4f} | {m_v2['macro_r']:<10.4f}")

    # 2. Per-class metrics
    print("\n" + "-" * 60)
    print("PER-CLASS P / R / F1")
    print("-" * 60)
    print(f"{'Class':<12} | {'LLM P/R/F1':<22} | {'v1 P/R/F1':<22} | {'v2 P/R/F1':<22}")
    print("-" * 84)
    for c in CLASSES:
        llm_str = f"{m_llm['per_class'][c]['p']:.2f} / {m_llm['per_class'][c]['r']:.2f} / {m_llm['per_class'][c]['f1']:.2f}"
        v1_str = f"{m_v1['per_class'][c]['p']:.2f} / {m_v1['per_class'][c]['r']:.2f} / {m_v1['per_class'][c]['f1']:.2f}"
        v2_str = f"{m_v2['per_class'][c]['p']:.2f} / {m_v2['per_class'][c]['r']:.2f} / {m_v2['per_class'][c]['f1']:.2f}"
        print(f"{c:<12} | {llm_str:<22} | {v1_str:<22} | {v2_str:<22}")

    # 3. Confusion matrices
    print("\n" + "-" * 60)
    print("CONFUSION MATRIX: LLM-only")
    print("-" * 60)
    print(format_cm(m_llm["cm"]))

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX: v1 Pipeline")
    print("-" * 60)
    print(format_cm(m_v1["cm"]))

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX: v2 Pipeline")
    print("-" * 60)
    print(format_cm(m_v2["cm"]))

    # 4. Paired table (only v1 right / only v2 right / both right / both wrong)
    only_v1_right = []
    only_v2_right = []
    both_right = []
    both_wrong = []

    for i in range(len(golds)):
        g = golds[i]
        v1_ok = (v1_preds[i] == g)
        v2_ok = (v2_preds[i] == g)
        info = {
            "text": res_df.loc[i, "text"],
            "gold": g,
            "v1_pred": v1_preds[i],
            "v1_src": res_df.loc[i, "v1_source"],
            "v2_pred": v2_preds[i],
            "v2_src": res_df.loc[i, "v2_source"],
            "v2_verdict": res_df.loc[i, "v2_verdict"],
        }
        if v1_ok and not v2_ok:
            only_v1_right.append(info)
        elif v2_ok and not v1_ok:
            only_v2_right.append(info)
        elif v1_ok and v2_ok:
            both_right.append(info)
        else:
            both_wrong.append(info)

    print("\n" + "=" * 60)
    print("PAIRED COMPARISON: v1 vs v2 (N=24)")
    print("=" * 60)
    print(f"Both right:    {len(both_right):>2} ({len(both_right)/n_rows*100:.1f}%)")
    print(f"Only v2 right: {len(only_v2_right):>2} ({len(only_v2_right)/n_rows*100:.1f}%)")
    print(f"Only v1 right: {len(only_v1_right):>2} ({len(only_v1_right)/n_rows*100:.1f}%)")
    print(f"Both wrong:    {len(both_wrong):>2} ({len(both_wrong)/n_rows*100:.1f}%)")

    if only_v2_right:
        print("\nRows where only v2 was right (v2 fixed v1 error):")
        for r in only_v2_right:
            print(f"- Gold: {r['gold']} | v1: {r['v1_pred']} ({r['v1_src']}) -> v2: {r['v2_pred']} ({r['v2_src']}, verdict={r['v2_verdict']})")
            print(f"  Text: {r['text'][:75]}")

    if only_v1_right:
        print("\nRows where only v1 was right (v2 regressed):")
        for r in only_v1_right:
            print(f"- Gold: {r['gold']} | v1: {r['v1_pred']} ({r['v1_src']}) -> v2: {r['v2_pred']} ({r['v2_src']}, verdict={r['v2_verdict']})")
            print(f"  Text: {r['text'][:75]}")


if __name__ == "__main__":
    main()
