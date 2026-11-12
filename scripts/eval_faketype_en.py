import os
import sys
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY set: {bool(groq_key)}")
if not groq_key:
    print("FATAL: GROQ_API_KEY is not set in environment! Aborting Groq steps loudly.")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reasoning.text_prep import prep_text
from reasoning.faketype_llm import classify
from reasoning.faketype_pipeline import classify_type, _predict_trained
from fact_check.fact_checker import fact_check

CLASSES = ["ordinary", "satire", "clickbait", "misleading"]

# In-memory claim cache for fact_check
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
    parser = argparse.ArgumentParser(description="Evaluate Fake-Type Classifiers")
    parser.add_argument("--test", action="store_true", help="Evaluate on faketype_test.csv instead of dev")
    args = parser.parse_args()

    name = "test" if args.test else "dev"
    csv_filename = f"faketype_{name}.csv"
    csv_path = PROJECT_ROOT / "data" / "eval" / csv_filename

    if not csv_path.exists():
        print(f"FATAL: Evaluation CSV {csv_path} does not exist!")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # Filter language == "en"
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == "en"].copy()

    # Clean gold labels
    df["gold"] = df["label"].apply(clean_label)
    total_rows = len(df)
    print(f"Loaded {total_rows} English rows from {csv_filename}.")

    # Model IDs for LLM 1 and LLM 2
    model_1 = os.getenv("GROQ_FAKETYPE_MODEL", "openai/gpt-oss-120b")
    model_2 = os.getenv("GROQ_FAKETYPE_MODEL_2", "qwen/qwen3.8-27b")
    print(f"Evaluating with LLM Model 1: {model_1}")
    print(f"Evaluating with LLM Model 2: {model_2}")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = results_dir / f"faketype_en_{name}.csv"

    # Incremental processing / resume
    existing_rows = {}
    if out_csv_path.exists():
        try:
            prev_df = pd.read_csv(out_csv_path)
            for idx, r in prev_df.iterrows():
                t = str(r["text"]).strip()
                existing_rows[t] = r.to_dict()
            print(f"Resuming: found {len(existing_rows)} rows already evaluated in {out_csv_path}.")
        except Exception as e:
            print(f"Warning reading existing results CSV: {e}")

    results_list = []

    for idx, row in df.iterrows():
        text = str(row["text"]).strip()
        gold = row["gold"]

        if text in existing_rows:
            results_list.append(existing_rows[text])
            continue

        print(f"[{len(results_list) + 1}/{total_rows}] Evaluating: {text[:60]}...")

        # 1. Trained-only (prep_text)
        t_label, t_conf = _predict_trained(text)

        # 2. Trained-only with punctuation stripped
        # replace non-alphanumeric characters except apostrophes with a space
        prepped = prep_text(text)
        punct_stripped = re.sub(r"[^\w\s']", " ", prepped)
        punct_stripped = re.sub(r"\s+", " ", punct_stripped).strip()
        t_nopunct_label, _ = _predict_trained(punct_stripped)

        # 3. LLM Model 1
        llm1_res = classify(text, model=model_1)
        llm1_label = llm1_res.get("label", "ordinary")
        llm1_conf = float(llm1_res.get("confidence", 0.0))
        llm1_cues = "; ".join(llm1_res.get("cues", []))

        # 4. LLM Model 2
        llm2_res = classify(text, model=model_2)
        llm2_label = llm2_res.get("label", "ordinary")
        llm2_conf = float(llm2_res.get("confidence", 0.0))

        # 5. Hybrid Pipeline
        pipe_res = classify_type(text, verifier_fn=cached_fact_check)
        pipe_label = pipe_res.get("label", "ordinary")
        pipe_source = pipe_res.get("source", "default")
        pipe_uncertain = pipe_res.get("uncertain", False)
        pipe_verdict = pipe_res.get("verdict", "")

        row_dict = {
            "text": text,
            "gold": gold,
            "pred_trained": t_label,
            "conf_trained": t_conf,
            "pred_trained_nopunct": t_nopunct_label,
            "pred_llm1": llm1_label,
            "conf_llm1": llm1_conf,
            "cues_llm1": llm1_cues,
            "pred_llm2": llm2_label,
            "conf_llm2": llm2_conf,
            "pred_pipeline": pipe_label,
            "source_pipeline": pipe_source,
            "uncertain_pipeline": pipe_uncertain,
            "verdict_pipeline": pipe_verdict,
        }

        results_list.append(row_dict)
        existing_rows[text] = row_dict

        # Write incrementally
        pd.DataFrame(results_list).to_csv(out_csv_path, index=False)

    eval_df = pd.DataFrame(results_list)

    # Write misclassified rows to errors file
    # results/faketype_en_errors_<name>.csv (text, gold, pred, trained label+conf, LLM label+cues)
    errors_df = eval_df[eval_df["pred_pipeline"] != eval_df["gold"]].copy()
    errors_df["trained label+conf"] = (
        errors_df["pred_trained"] + " (" + errors_df["conf_trained"].round(3).astype(str) + ")"
    )
    errors_df["LLM label+cues"] = (
        errors_df["pred_llm1"] + " [" + errors_df["cues_llm1"] + "]"
    )
    errors_out = errors_df[["text", "gold", "pred_pipeline", "trained label+conf", "LLM label+cues"]].rename(
        columns={"pred_pipeline": "pred"}
    )
    errors_path = results_dir / f"faketype_en_errors_{name}.csv"
    errors_out.to_csv(errors_path, index=False)
    print(f"Wrote {len(errors_out)} errors to {errors_path}.")

    # --- PRINT REQUIRED TABLES AND METRICS ---
    golds = eval_df["gold"].tolist()

    m_trained = compute_metrics(golds, eval_df["pred_trained"].tolist())
    m_llm1 = compute_metrics(golds, eval_df["pred_llm1"].tolist())
    m_pipe = compute_metrics(golds, eval_df["pred_pipeline"].tolist())

    print("\n" + "=" * 60)
    print(f"1. TRAINED-ONLY EVALUATION ({name.upper()})")
    print(f"Accuracy: {m_trained['accuracy']:.4f} | Macro-F1: {m_trained['macro_f1']:.4f}")
    print("Per-Class Metrics:")
    for c in CLASSES:
        st = m_trained['per_class'][c]
        print(f"  {c:<12}: P={st['p']:.4f}, R={st['r']:.4f}, F1={st['f1']:.4f}")
    print("Confusion Matrix:")
    print(format_cm(m_trained['cm']))

    print("\n" + "=" * 60)
    print(f"2. LLM-ONLY (MODEL 1: {model_1}) EVALUATION ({name.upper()})")
    print(f"Accuracy: {m_llm1['accuracy']:.4f} | Macro-F1: {m_llm1['macro_f1']:.4f}")
    print("Per-Class Metrics:")
    for c in CLASSES:
        st = m_llm1['per_class'][c]
        print(f"  {c:<12}: P={st['p']:.4f}, R={st['r']:.4f}, F1={st['f1']:.4f}")
    print("Confusion Matrix:")
    print(format_cm(m_llm1['cm']))

    print("\n" + "=" * 60)
    print(f"3. HYBRID PIPELINE EVALUATION ({name.upper()})")
    print(f"Accuracy: {m_pipe['accuracy']:.4f} | Macro-F1: {m_pipe['macro_f1']:.4f}")
    print("Per-Class Metrics:")
    for c in CLASSES:
        st = m_pipe['per_class'][c]
        print(f"  {c:<12}: P={st['p']:.4f}, R={st['r']:.4f}, F1={st['f1']:.4f}")
    print("Confusion Matrix:")
    print(format_cm(m_pipe['cm']))

    print("\n" + "=" * 60)
    print("4. ADDITIONAL DIAGNOSTICS & CHECKS")

    # (a) Accuracy where trained and LLM agree vs disagree
    agree_mask = eval_df["pred_trained"] == eval_df["pred_llm1"]
    acc_agree = accuracy_score(eval_df.loc[agree_mask, "gold"], eval_df.loc[agree_mask, "pred_pipeline"]) if agree_mask.sum() > 0 else 0.0
    acc_disagree = accuracy_score(eval_df.loc[~agree_mask, "gold"], eval_df.loc[~agree_mask, "pred_pipeline"]) if (~agree_mask).sum() > 0 else 0.0
    print(f"Trained vs LLM Agreement: {agree_mask.sum()}/{len(eval_df)} rows ({agree_mask.mean():.4f})")
    print(f"Pipeline Accuracy when Agreed:    {acc_agree:.4f} (n={agree_mask.sum()})")
    print(f"Pipeline Accuracy when Disagreed: {acc_disagree:.4f} (n={(~agree_mask).sum()})")

    # (b) Trained-only accuracy on rows with vs without a digit
    has_digit = eval_df["text"].apply(lambda t: bool(re.search(r"\d", t)))
    acc_digit = accuracy_score(eval_df.loc[has_digit, "gold"], eval_df.loc[has_digit, "pred_trained"]) if has_digit.sum() > 0 else 0.0
    acc_nodigit = accuracy_score(eval_df.loc[~has_digit, "gold"], eval_df.loc[~has_digit, "pred_trained"]) if (~has_digit).sum() > 0 else 0.0
    print(f"\nTrained-only Accuracy with Digit:    {acc_digit:.4f} (n={has_digit.sum()})")
    print(f"Trained-only Accuracy without Digit: {acc_nodigit:.4f} (n={(~has_digit).sum()})")

    # (c) Trained-only predictions: prep_text vs punctuation additionally stripped
    punct_agree = (eval_df["pred_trained"] == eval_df["pred_trained_nopunct"]).mean()
    acc_nopunct = accuracy_score(eval_df["gold"], eval_df["pred_trained_nopunct"])
    print(f"\nPunctuation Shortcut Check:")
    print(f"Agreement (prep_text vs punct-stripped): {punct_agree:.4f}")
    print(f"Trained Accuracy with prep_text:         {m_trained['accuracy']:.4f}")
    print(f"Trained Accuracy with punct-stripped:    {acc_nopunct:.4f}")

    # (d) Agreement rate between LLM model 1 and model 2
    llm_agree = (eval_df["pred_llm1"] == eval_df["pred_llm2"]).mean()
    print(f"\nLLM Agreement Rate (Model 1 vs Model 2): {llm_agree:.4f} ({sum(eval_df['pred_llm1'] == eval_df['pred_llm2'])}/{len(eval_df)})")


if __name__ == "__main__":
    main()
