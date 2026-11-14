import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fact_check.fact_checker import retrieve_evidence

DEBUG_IDS = {
    "T01", "T02", "T03", "T04", "T05", "T06",
    "F01", "F02", "F03", "F04", "F05", "F06",
}

CLASSES = ["TRUE", "FALSE", "UNVERIFIABLE"]
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]


def wilson_ci(k, n, z=1.95996398454):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    spread = (z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * (n ** 2)))) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


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
        print(f"FATAL: {ablation_csv} not found!")
        sys.exit(1)

    df = pd.read_csv(ablation_csv)
    # Filter to REPORT set (exclude DEBUG rows)
    report_df = df[~df["id"].isin(DEBUG_IDS)].copy().reset_index(drop=True)
    n = len(report_df)
    print(f"Total rows: {len(df)} | REPORT set rows: {n}")

    golds = [clean_v(g) for g in report_df["gold"]]

    # ---------------------------------------------------------
    # a) Per config metrics
    # ---------------------------------------------------------
    config_metrics = {}
    for c in CONFIGS:
        preds = [clean_v(p) for p in report_df[f"{c.lower()}_verdict"]]
        k = sum(1 for g, p in zip(golds, preds) if g == p)
        acc = k / n
        ci_low, ci_high = wilson_ci(k, n)

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            golds, preds, labels=CLASSES, average="macro", zero_division=0
        )
        p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
            golds, preds, labels=CLASSES, average=None, zero_division=0
        )
        cm = confusion_matrix(golds, preds, labels=CLASSES)

        unv_count = sum(1 for p in preds if p == "UNVERIFIABLE")
        share_unv = unv_count / n

        answered = [i for i in range(n) if preds[i] in ("TRUE", "FALSE")]
        if answered:
            acc_ans = sum(1 for i in answered if preds[i] == golds[i]) / len(answered)
        else:
            acc_ans = 0.0

        config_metrics[c] = {
            "acc": acc,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "macro_f1": f1_macro,
            "p_macro": p_macro,
            "r_macro": r_macro,
            "p_cls": p_cls,
            "r_cls": r_cls,
            "f1_cls": f1_cls,
            "cm": cm,
            "share_unv": share_unv,
            "acc_ans": acc_ans,
            "preds": preds,
        }

    # ---------------------------------------------------------
    # b) Category & Language breakdown
    # ---------------------------------------------------------
    categories = ["stable_fact", "wrong_fact", "invented", "recent_news", "recent_news_mutated"]
    cat_metrics = {}
    for cat in categories:
        cat_sub = report_df[report_df["category"] == cat]
        cat_n = len(cat_sub)
        cat_golds = [clean_v(g) for g in cat_sub["gold"]]
        cat_accs = {}
        for c in CONFIGS:
            cat_preds = [clean_v(p) for p in cat_sub[f"{c.lower()}_verdict"]]
            cat_k = sum(1 for g, p in zip(cat_golds, cat_preds) if g == p)
            cat_accs[c] = (cat_k / cat_n) if cat_n > 0 else 0.0
        cat_metrics[cat] = {"n": cat_n, "accs": cat_accs}

    # Language
    lang_metrics = {}
    en_sub = report_df[report_df["language"] == "en"]
    non_en_sub = report_df[report_df["language"] != "en"]
    for l_name, sub in [("en", en_sub), ("non-en", non_en_sub)]:
        l_n = len(sub)
        l_golds = [clean_v(g) for g in sub["gold"]]
        l_accs = {}
        for c in CONFIGS:
            l_preds = [clean_v(p) for p in sub[f"{c.lower()}_verdict"]]
            l_k = sum(1 for g, p in zip(l_golds, l_preds) if g == p)
            l_accs[c] = (l_k / l_n) if l_n > 0 else 0.0
        lang_metrics[l_name] = {"n": l_n, "accs": l_accs}

    # ---------------------------------------------------------
    # c) Paired comparisons
    # ---------------------------------------------------------
    pairs = [
        ("C4", "C1"),
        ("C4", "C0"),
        ("C4", "C5"),
        ("C3", "C2"),
        ("C2", "C1"),
    ]
    pair_metrics = {}
    for a, b in pairs:
        preds_a = config_metrics[a]["preds"]
        preds_b = config_metrics[b]["preds"]
        only_a = sum(1 for i in range(n) if preds_a[i] == golds[i] and preds_b[i] != golds[i])
        only_b = sum(1 for i in range(n) if preds_b[i] == golds[i] and preds_a[i] != golds[i])
        both_r = sum(1 for i in range(n) if preds_a[i] == golds[i] and preds_b[i] == golds[i])
        both_w = sum(1 for i in range(n) if preds_a[i] != golds[i] and preds_b[i] != golds[i])
        pair_metrics[f"{a} vs {b}"] = {
            "only_a": only_a,
            "only_b": only_b,
            "both_r": both_r,
            "both_w": both_w,
        }

    # ---------------------------------------------------------
    # d) C4 statistics
    # ---------------------------------------------------------
    c4_sources = report_df["c4_verdict_source"].value_counts().to_dict()
    c4_conflict_count = int(report_df["c4_conflict"].sum())
    secs = report_df["c4_seconds"].dropna().astype(float)
    c4_med_sec = float(np.median(secs)) if len(secs) > 0 else 0.0
    c4_p90_sec = float(np.percentile(secs, 90)) if len(secs) > 0 else 0.0

    # ---------------------------------------------------------
    # e) Error analysis CSV (C4 wrong on REPORT set)
    # ---------------------------------------------------------
    errors = []
    err_csv = PROJECT_ROOT / "results" / "verifier_errors.csv"
    for idx, row in report_df.iterrows():
        g = clean_v(row["gold"])
        c4_v = clean_v(row["c4_verdict"])
        if c4_v != g:
            claim_n = row["claim_n"]
            # Retrieve cached top-2 evidence
            ev = retrieve_evidence(claim_n, max_results=2)
            t1 = ev[0].get("title", "") if len(ev) > 0 else ""
            u1 = ev[0].get("url", "") if len(ev) > 0 else ""
            t2 = ev[1].get("title", "") if len(ev) > 1 else ""
            u2 = ev[1].get("url", "") if len(ev) > 1 else ""
            errors.append({
                "id": row["id"],
                "claim_n": claim_n,
                "gold": g,
                "c4_verdict": c4_v,
                "c4_verdict_source": row["c4_verdict_source"],
                "top_evidence_title_1": t1,
                "top_evidence_url_1": u1,
                "top_evidence_title_2": t2,
                "top_evidence_url_2": u2,
                "c0_verdict": clean_v(row["c0_verdict"]),
            })
    pd.DataFrame(errors).to_csv(err_csv, index=False)
    print(f"Written {len(errors)} error rows to {err_csv.name}.")

    # ---------------------------------------------------------
    # Format results/verifier_ablation_summary.txt (tables only)
    # ---------------------------------------------------------
    summary_path = PROJECT_ROOT / "results" / "verifier_ablation_summary.txt"
    lines = []

    # Table A: Overall Performance
    lines.append("TABLE A: OVERALL PERFORMANCE ON REPORT SET (N=37)")
    lines.append(f"{'Config':<8} | {'Accuracy (95% CI)':<22} | {'Macro-F1':<10} | {'% UNVERIF':<11} | {'Acc (Ans)':<10}")
    lines.append("-" * 70)
    for c in CONFIGS:
        m = config_metrics[c]
        ci_str = f"{m['acc']:.4f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
        lines.append(f"{c:<8} | {ci_str:<22} | {m['macro_f1']:<10.4f} | {m['share_unv']*100:<10.1f}% | {m['acc_ans']:<10.4f}")

    # Table A2: Per-Class P / R / F1
    lines.append("\nTABLE A2: PER-CLASS P / R / F1 (T / F / U)")
    lines.append(f"{'Config':<8} | {'TRUE P/R/F1':<22} | {'FALSE P/R/F1':<22} | {'UNVERIFIABLE P/R/F1':<22}")
    lines.append("-" * 80)
    for c in CONFIGS:
        m = config_metrics[c]
        t_str = f"{m['p_cls'][0]:.2f} / {m['r_cls'][0]:.2f} / {m['f1_cls'][0]:.2f}"
        f_str = f"{m['p_cls'][1]:.2f} / {m['r_cls'][1]:.2f} / {m['f1_cls'][1]:.2f}"
        u_str = f"{m['p_cls'][2]:.2f} / {m['r_cls'][2]:.2f} / {m['f1_cls'][2]:.2f}"
        lines.append(f"{c:<8} | {t_str:<22} | {f_str:<22} | {u_str:<22}")

    # Table A3: 3x3 Confusion Matrices
    lines.append("\nTABLE A3: 3x3 CONFUSION MATRICES [Rows=Gold T/F/U, Cols=Pred T/F/U]")
    lines.append(f"{'Config':<8} | {'Gold TRUE':<16} | {'Gold FALSE':<16} | {'Gold UNVERIFIABLE':<16}")
    lines.append("-" * 65)
    for c in CONFIGS:
        cm = config_metrics[c]["cm"]
        r0 = f"[{cm[0][0]}, {cm[0][1]}, {cm[0][2]}]"
        r1 = f"[{cm[1][0]}, {cm[1][1]}, {cm[1][2]}]"
        r2 = f"[{cm[2][0]}, {cm[2][1]}, {cm[2][2]}]"
        lines.append(f"{c:<8} | {r0:<16} | {r1:<16} | {r2:<16}")

    # Table B: Category Breakdown
    lines.append("\nTABLE B: ACCURACY BY CATEGORY")
    lines.append(f"{'Category':<22} | {'n':<4} | " + " | ".join(f"{c:<6}" for c in CONFIGS))
    lines.append("-" * 70)
    for cat in categories:
        info = cat_metrics[cat]
        cat_acc_strs = " | ".join(f"{info['accs'][c]:<6.3f}" for c in CONFIGS)
        lines.append(f"{cat:<22} | {info['n']:<4} | {cat_acc_strs}")

    # Table B2: Language Breakdown
    lines.append("\nTABLE B2: ACCURACY BY LANGUAGE")
    lines.append(f"{'Language':<10} | {'n':<4} | " + " | ".join(f"{c:<6}" for c in CONFIGS))
    lines.append("-" * 60)
    for l_name in ["en", "non-en"]:
        info = lang_metrics[l_name]
        l_acc_strs = " | ".join(f"{info['accs'][c]:<6.3f}" for c in CONFIGS)
        lines.append(f"{l_name:<10} | {info['n']:<4} | {l_acc_strs}")

    # Table C: Paired Comparisons
    lines.append("\nTABLE C: PAIRED COMPARISONS")
    lines.append(f"{'Comparison':<16} | {'Only A Right':<14} | {'Only B Right':<14} | {'Both Right':<12} | {'Both Wrong':<12}")
    lines.append("-" * 75)
    for pair_name, p in pair_metrics.items():
        lines.append(f"{pair_name:<16} | {p['only_a']:<14} | {p['only_b']:<14} | {p['both_r']:<12} | {p['both_w']:<12}")

    # Table D: C4 Diagnostics
    lines.append("\nTABLE D: C4 FULL SYSTEM DIAGNOSTICS")
    lines.append(f"{'Metric':<30} | {'Value':<15}")
    lines.append("-" * 48)
    lines.append(f"{'Rule-based verdicts':<30} | {c4_sources.get('rule_based', 0):<15}")
    lines.append(f"{'LLM fallback verdicts':<30} | {c4_sources.get('llm_fallback', 0):<15}")
    lines.append(f"{'Conflict count':<30} | {c4_conflict_count:<15}")
    lines.append(f"{'Median latency (seconds)':<30} | {c4_med_sec:<15.2f}")
    lines.append(f"{'P90 latency (seconds)':<30} | {c4_p90_sec:<15.2f}")

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
