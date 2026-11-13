import os
import sys
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(PROJECT_ROOT, "results")

dev_path = os.path.join(results_dir, "faketype_en_dev.csv")
test_path = os.path.join(results_dir, "faketype_en_test.csv")

if not os.path.exists(dev_path) or not os.path.exists(test_path):
    print("FATAL: Both dev and test results CSVs must exist!")
    sys.exit(1)

dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)
pooled_df = pd.concat([dev_df, test_df], ignore_index=True)

CLASSES = ["ordinary", "satire", "clickbait", "misleading"]
CLASSES_3 = ["plain", "satire", "clickbait"]


def format_cm(cm, labels):
    header = f"{'Gold \\ Pred':<14} | " + " | ".join(f"{c:<10}" for c in labels)
    sep = "-" * len(header)
    lines = [header, sep]
    for i, c in enumerate(labels):
        row_str = f"{c:<14} | " + " | ".join(f"{cm[i][j]:<10}" for j in range(len(labels)))
        lines.append(row_str)
    return "\n".join(lines)


def wilson_ci(k, n, z=1.95996):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z**2) / n
    centre = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z**2) / (4 * (n**2))))
    return max(0.0, centre - margin), min(1.0, centre + margin)


# a) Trained-only collapsed to 3 classes (gold ordinary+misleading -> plain, predicted ordinary -> plain)
def collapse_3(gold_col, pred_col):
    gold_3 = [
        "plain" if g in ("ordinary", "misleading") else g for g in gold_col
    ]
    pred_3 = [
        "plain" if p in ("ordinary", "plain") else p for p in pred_col
    ]
    acc = accuracy_score(gold_3, pred_3)
    f1 = precision_recall_fscore_support(
        gold_3, pred_3, labels=CLASSES_3, average="macro", zero_division=0
    )[2]
    return acc, f1


acc_dev_3, f1_dev_3 = collapse_3(dev_df["gold"], dev_df["pred_trained"])
acc_test_3, f1_test_3 = collapse_3(test_df["gold"], test_df["pred_trained"])
acc_pool_3, f1_pool_3 = collapse_3(pooled_df["gold"], pooled_df["pred_trained"])

# b) Pooled dev+test for all three methods
methods = [
    ("Trained-only", "pred_trained"),
    ("LLM-only (Model 1)", "pred_llm1"),
    ("Hybrid Pipeline", "pred_pipeline"),
]

pooled_metrics = {}
for name, col in methods:
    golds = pooled_df["gold"].tolist()
    preds = pooled_df[col].tolist()
    acc = accuracy_score(golds, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        golds, preds, labels=CLASSES, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        golds, preds, labels=CLASSES, average=None, zero_division=0
    )
    cm = confusion_matrix(golds, preds, labels=CLASSES)
    pooled_metrics[name] = {
        "acc": acc,
        "f1": f1_macro,
        "p_cls": p_cls,
        "r_cls": r_cls,
        "f1_cls": f1_cls,
        "cm": cm,
    }

# c) 95% Wilson confidence interval for accuracy
ci_test = {}
ci_pool = {}
for name, col in methods:
    k_test = (test_df["gold"] == test_df[col]).sum()
    n_test = len(test_df)
    ci_test[name] = wilson_ci(k_test, n_test)

    k_pool = (pooled_df["gold"] == pooled_df[col]).sum()
    n_pool = len(pooled_df)
    ci_pool[name] = wilson_ci(k_pool, n_pool)

# d) Paired comparison hybrid vs LLM-only on pooled rows
hyb_right = (pooled_df["pred_pipeline"] == pooled_df["gold"])
llm_right = (pooled_df["pred_llm1"] == pooled_df["gold"])

only_hyb = (hyb_right & ~llm_right).sum()
only_llm = (~hyb_right & llm_right).sum()
both_right = (hyb_right & llm_right).sum()
both_wrong = (~hyb_right & ~llm_right).sum()

# Construct final report text
out_lines = []
out_lines.append("=" * 60)
out_lines.append("FAKETYPE ENGLISH EVALUATION: EXTRA TABLES & POOLED ANALYSIS")
out_lines.append("=" * 60)

out_lines.append("\nA) TRAINED-ONLY COLLAPSED TO 3 CLASSES (plain, satire, clickbait):")
out_lines.append("Split      Accuracy    Macro-F1    N")
out_lines.append("-" * 45)
out_lines.append(f"Dev        {acc_dev_3:<11.4f} {f1_dev_3:<11.4f} {len(dev_df)}")
out_lines.append(f"Test       {acc_test_3:<11.4f} {f1_test_3:<11.4f} {len(test_df)}")
out_lines.append(f"Pooled     {acc_pool_3:<11.4f} {f1_pool_3:<11.4f} {len(pooled_df)}")

out_lines.append("\nB) POOLED DEV+TEST (N=76) EVALUATION FOR ALL THREE METHODS:")
for name, m in pooled_metrics.items():
    out_lines.append(f"\n--- {name} (Pooled N=76) ---")
    out_lines.append(f"Accuracy: {m['acc']:.4f} | Macro-F1: {m['f1']:.4f}")
    out_lines.append("Per-Class Metrics:")
    for i, c in enumerate(CLASSES):
        out_lines.append(f"  {c:<12}: P={m['p_cls'][i]:.4f}, R={m['r_cls'][i]:.4f}, F1={m['f1_cls'][i]:.4f}")
    out_lines.append("Confusion Matrix:")
    out_lines.append(format_cm(m['cm'], CLASSES))

out_lines.append("\nC) 95% WILSON SCORE CONFIDENCE INTERVALS FOR ACCURACY:")
out_lines.append(f"{'Method':<22} | {'Test (n=48) Acc [95% CI]':<32} | {'Pooled (n=76) Acc [95% CI]':<32}")
out_lines.append("-" * 92)
for name, col in methods:
    acc_t = (test_df["gold"] == test_df[col]).mean()
    l_t, u_t = ci_test[name]
    acc_p = (pooled_df["gold"] == pooled_df[col]).mean()
    l_p, u_p = ci_pool[name]
    out_lines.append(f"{name:<22} | {acc_t:.4f} [{l_t:.4f}, {u_t:.4f}]             | {acc_p:.4f} [{l_p:.4f}, {u_p:.4f}]")

out_lines.append("\nD) PAIRED COMPARISON: HYBRID vs LLM-ONLY (POOLED N=76):")
out_lines.append(f"Both correct:          {both_right} ({both_right/len(pooled_df)*100:.1f}%)")
out_lines.append(f"Only Hybrid correct:   {only_hyb} ({only_hyb/len(pooled_df)*100:.1f}%)")
out_lines.append(f"Only LLM correct:      {only_llm} ({only_llm/len(pooled_df)*100:.1f}%)")
out_lines.append(f"Both wrong:            {both_wrong} ({both_wrong/len(pooled_df)*100:.1f}%)")

final_text = "\n".join(out_lines) + "\n"
out_txt_path = os.path.join(results_dir, "faketype_en_final.txt")
with open(out_txt_path, "w", encoding="utf-8") as f:
    f.write(final_text)

print(final_text)
print(f"Written results to {out_txt_path}")
