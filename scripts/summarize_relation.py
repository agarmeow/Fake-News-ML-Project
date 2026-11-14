import os
import sys
import math
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSES = ["TRUE", "FALSE", "UNVERIFIABLE"]
CONFIGS = ["C2", "C3", "C4"]


def wilson_ci(k, n, z=1.95996398454):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    spread = (z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * (n ** 2)))) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def clean_v(v):
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
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
    csv_path = PROJECT_ROOT / "results" / "verifier_relation.csv"
    if not csv_path.exists():
        print(f"FATAL: {csv_path} not found!")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"Loaded {n} relation rows.")

    golds = [clean_v(g) for g in df["gold"]]
    df["gold_clean"] = golds

    lines = []

    # 1. Per config performance
    lines.append("TABLE 1: PERFORMANCE ON RELATION CLAIMS (N=12)")
    lines.append(f"{'Config':<8} | {'Accuracy (95% Wilson CI)':<26} | {'% UNVERIFIABLE':<16} | {'Correct/Total':<14}")
    lines.append("-" * 70)

    config_data = {}
    for c in CONFIGS:
        preds = [clean_v(p) for p in df[f"{c.lower()}_verdict"]]
        k = sum(1 for g, p in zip(golds, preds) if g == p)
        acc = k / n
        ci_l, ci_h = wilson_ci(k, n)
        unv = sum(1 for p in preds if p == "UNVERIFIABLE")
        pct_unv = unv / n
        config_data[c] = {
            "preds": preds,
            "acc": acc,
            "ci_l": ci_l,
            "ci_h": ci_h,
            "pct_unv": pct_unv,
            "k": k,
            "cm": confusion_matrix(golds, preds, labels=CLASSES),
        }
        ci_str = f"{acc:.4f} [{ci_l:.3f}, {ci_h:.3f}]"
        lines.append(f"{c:<8} | {ci_str:<26} | {pct_unv*100:<15.1f}% | {f'{k}/{n}':<14}")

    # 2. 3x3 Confusion Matrices
    lines.append("\nTABLE 2: 3x3 CONFUSION MATRICES [Rows=Gold T/F/U, Cols=Pred T/F/U]")
    lines.append(f"{'Config':<8} | {'Gold TRUE':<16} | {'Gold FALSE':<16} | {'Gold UNVERIFIABLE':<16}")
    lines.append("-" * 65)
    for c in CONFIGS:
        cm = config_data[c]["cm"]
        r0 = f"[{cm[0][0]}, {cm[0][1]}, {cm[0][2]}]"
        r1 = f"[{cm[1][0]}, {cm[1][1]}, {cm[1][2]}]"
        r2 = f"[{cm[2][0]}, {cm[2][1]}, {cm[2][2]}]"
        lines.append(f"{c:<8} | {r0:<16} | {r1:<16} | {r2:<16}")

    # 3. Paired C3 vs C2
    p_c2 = config_data["C2"]["preds"]
    p_c3 = config_data["C3"]["preds"]
    only_c3 = sum(1 for i in range(n) if p_c3[i] == golds[i] and p_c2[i] != golds[i])
    only_c2 = sum(1 for i in range(n) if p_c2[i] == golds[i] and p_c3[i] != golds[i])
    both_r = sum(1 for i in range(n) if p_c3[i] == golds[i] and p_c2[i] == golds[i])
    both_w = sum(1 for i in range(n) if p_c3[i] != golds[i] and p_c2[i] != golds[i])

    lines.append("\nTABLE 3: PAIRED COMPARISON: C3 vs C2 (N=12)")
    lines.append(f"{'Only C3 Right':<16} | {'Only C2 Right':<16} | {'Both Right':<12} | {'Both Wrong':<12}")
    lines.append("-" * 60)
    lines.append(f"{only_c3:<16} | {only_c2:<16} | {both_r:<12} | {both_w:<12}")

    # 4. Rows where C3 differs from C2
    lines.append("\nTABLE 4: ROWS WHERE C3 DIFFERS FROM C2")
    lines.append(f"{'ID':<6} | {'Gold':<8} | {'C2 Verdict':<14} | {'C3 Verdict':<14} | {'Claim':<45}")
    lines.append("-" * 92)
    diff_count = 0
    for idx, row in df.iterrows():
        g = clean_v(row["gold"])
        v2 = clean_v(row["c2_verdict"])
        v3 = clean_v(row["c3_verdict"])
        if v2 != v3:
            diff_count += 1
            lines.append(f"{row['id']:<6} | {g:<8} | {v2:<14} | {v3:<14} | {str(row['claim_n'])[:43]:<45}")

    if diff_count == 0:
        lines.append("None (C2 and C3 produced identical verdicts on all rows)")

    out_text = "\n".join(lines)
    summary_path = PROJECT_ROOT / "results" / "verifier_relation_summary.txt"
    summary_path.write_text(out_text, encoding="utf-8")
    print("\n" + out_text)


if __name__ == "__main__":
    main()
