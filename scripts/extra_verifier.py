import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEBUG_IDS = {
    "T01", "T02", "T03", "T04", "T05", "T06",
    "F01", "F02", "F03", "F04", "F05", "F06",
}

CLASSES = ["TRUE", "FALSE", "UNVERIFIABLE"]


def clean_v(v):
    if not isinstance(v, str):
        return "UNVERIFIABLE"
    vu = v.strip().upper()
    if vu in ("TRUE", "FALSE", "UNVERIFIABLE"):
        return vu
    if "TRUE" in vu:
        return "TRUE"
    if "FALSE" in vu:
        return "FALSE"
    return "UNVERIFIABLE"


def main():
    ablation_csv = PROJECT_ROOT / "results" / "verifier_ablation.csv"
    if not ablation_csv.exists():
        print(f"FATAL: {ablation_csv} does not exist!")
        sys.exit(1)

    df = pd.read_csv(ablation_csv)
    rep = df[~df["id"].isin(DEBUG_IDS)].copy().reset_index(drop=True)

    golds = [clean_v(g) for g in rep["gold"]]
    c0_preds = [clean_v(p) for p in rep["c0_verdict"]]
    c4_preds = [clean_v(p) for p in rep["c4_verdict"]]
    c5_preds = [clean_v(p) for p in rep["c5_verdict"]]
    rep["gold_clean"] = golds
    rep["c0_clean"] = c0_preds
    rep["c4_clean"] = c4_preds
    rep["c5_clean"] = c5_preds

    lines = []

    # a) C4 accuracy and n by verdict_source (rule_based vs llm_fallback); same split for C5 on those rows
    lines.append("TABLE A: C4 AND C5 ACCURACY BY VERDICT_SOURCE (REPORT SET, N=37)")
    lines.append(f"{'Verdict Source':<20} | {'n':<4} | {'C4 Correct':<10} | {'C4 Accuracy':<12} | {'C5 Correct':<10} | {'C5 Accuracy':<12}")
    lines.append("-" * 80)
    for src in ["rule_based", "llm_fallback"]:
        sub = rep[rep["c4_verdict_source"] == src]
        sn = len(sub)
        c4_corr = sum(1 for _, r in sub.iterrows() if r["c4_clean"] == r["gold_clean"])
        c5_corr = sum(1 for _, r in sub.iterrows() if r["c5_clean"] == r["gold_clean"])
        c4_acc = (c4_corr / sn) if sn > 0 else 0.0
        c5_acc = (c5_corr / sn) if sn > 0 else 0.0
        lines.append(f"{src:<20} | {sn:<4} | {c4_corr:<10} | {c4_acc:<12.4f} | {c5_corr:<10} | {c5_acc:<12.4f}")

    # b) Accuracy by language group (en vs non-en) for C0, C4, C5, with n
    lines.append("\nTABLE B: ACCURACY BY LANGUAGE GROUP")
    lines.append(f"{'Language Group':<16} | {'n':<4} | {'C0 Accuracy':<12} | {'C4 Accuracy':<12} | {'C5 Accuracy':<12}")
    lines.append("-" * 66)
    en_sub = rep[rep["language"] == "en"]
    non_en_sub = rep[rep["language"] != "en"]
    for l_name, sub in [("en", en_sub), ("non-en", non_en_sub)]:
        sn = len(sub)
        c0_acc = sum(1 for _, r in sub.iterrows() if r["c0_clean"] == r["gold_clean"]) / sn if sn > 0 else 0.0
        c4_acc = sum(1 for _, r in sub.iterrows() if r["c4_clean"] == r["gold_clean"]) / sn if sn > 0 else 0.0
        c5_acc = sum(1 for _, r in sub.iterrows() if r["c5_clean"] == r["gold_clean"]) / sn if sn > 0 else 0.0
        lines.append(f"{l_name:<16} | {sn:<4} | {c0_acc:<12.4f} | {c4_acc:<12.4f} | {c5_acc:<12.4f}")

    # c) Per category, number correct / n for C0, C4, C5
    lines.append("\nTABLE C: NUMBER CORRECT / N BY CATEGORY")
    lines.append(f"{'Category':<22} | {'n':<4} | {'C0 (Corr/n)':<12} | {'C4 (Corr/n)':<12} | {'C5 (Corr/n)':<12}")
    lines.append("-" * 72)
    categories = ["stable_fact", "wrong_fact", "invented", "recent_news", "recent_news_mutated"]
    for cat in categories:
        sub = rep[rep["category"] == cat]
        sn = len(sub)
        c0_k = sum(1 for _, r in sub.iterrows() if r["c0_clean"] == r["gold_clean"])
        c4_k = sum(1 for _, r in sub.iterrows() if r["c4_clean"] == r["gold_clean"])
        c5_k = sum(1 for _, r in sub.iterrows() if r["c5_clean"] == r["gold_clean"])
        lines.append(f"{cat:<22} | {sn:<4} | {f'{c0_k}/{sn}':<12} | {f'{c4_k}/{sn}':<12} | {f'{c5_k}/{sn}':<12}")

    # d) C4 errors: id, gold, C4 verdict, verdict_source, C0 verdict, C5 verdict
    lines.append("\nTABLE D: C4 ERRORS (REPORT SET)")
    lines.append(f"{'ID':<6} | {'Gold':<8} | {'C4 Verdict':<12} | {'Verdict Source':<15} | {'C0 Verdict':<12} | {'C5 Verdict':<12}")
    lines.append("-" * 75)
    c4_errs = rep[rep["c4_clean"] != rep["gold_clean"]]
    for _, r in c4_errs.iterrows():
        lines.append(f"{r['id']:<6} | {r['gold_clean']:<8} | {r['c4_clean']:<12} | {r['c4_verdict_source']:<15} | {r['c0_clean']:<12} | {r['c5_clean']:<12}")

    # e) Rows where the C4 conflict flag is set: id, gold, rule-based verdict, LLM verdict
    lines.append("\nTABLE E: C4 CONFLICT ROWS")
    lines.append(f"{'ID':<6} | {'Gold':<8} | {'Rule-based Verdict':<20} | {'LLM Verdict':<15}")
    lines.append("-" * 55)
    conflict_rows = rep[rep["c4_conflict"] == True]
    for _, r in conflict_rows.iterrows():
        # In fact_check, conflict is set when rule-based verdict in {TRUE, FALSE} and LLM returned opposite
        rule_v = "TRUE" if r["c4_clean"] == "TRUE" else "FALSE"
        llm_v = "FALSE" if rule_v == "TRUE" else "TRUE"
        lines.append(f"{r['id']:<6} | {r['gold_clean']:<8} | {rule_v:<20} | {llm_v:<15}")

    # f) For the 8 recent-news rows only: C0, C1, C4, C5 verdict next to the gold label
    lines.append("\nTABLE F: RECENT-NEWS ROWS (N=8)")
    lines.append(f"{'ID':<6} | {'Gold':<8} | {'C0 Verdict':<12} | {'C1 Verdict':<12} | {'C4 Verdict':<12} | {'C5 Verdict':<12}")
    lines.append("-" * 72)
    rn_sub = rep[rep["category"].str.startswith("recent_news")]
    for _, r in rn_sub.iterrows():
        c1_v = clean_v(r["c1_verdict"])
        lines.append(f"{r['id']:<6} | {r['gold_clean']:<8} | {r['c0_clean']:<12} | {c1_v:<12} | {r['c4_clean']:<12} | {r['c5_clean']:<12}")

    out_text = "\n".join(lines)
    out_path = PROJECT_ROOT / "results" / "verifier_extra.txt"
    out_path.write_text(out_text, encoding="utf-8")
    print(out_text)


if __name__ == "__main__":
    main()
