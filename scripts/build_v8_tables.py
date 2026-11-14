import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import binomtest

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)

def clean_v(v):
    if pd.isna(v):
        return "UNVERIFIABLE"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    v = str(v).strip().upper()
    return v if v in ["TRUE", "FALSE", "UNVERIFIABLE"] else "UNVERIFIABLE"

# Load ablation data
df_abl = pd.read_csv("results/verifier_ablation.csv")
exclude = [f"T{i:02d}" for i in range(1, 7)] + [f"F{i:02d}" for i in range(1, 7)]
df = df_abl[~df_abl["id"].isin(exclude)].copy()
df["gold"] = df["gold"].apply(clean_v)

for col in ["c0_verdict", "c1_verdict", "c2_verdict", "c3_verdict", "c4_verdict", "c5_verdict"]:
    df[col] = df[col].apply(clean_v)

# Load rerun data to update C4
df_rerun = pd.read_csv("results/verifier_rerun.csv")
rerun_map = {}
for _, r in df_rerun.iterrows():
    if r["file"] == "verifier_ablation.csv":
        rerun_map[r["id"]] = {
            "verdict": clean_v(r["after_verdict"]),
            "source": r["after_source"],
            "llm_status": r["llm_status"],
            "llm_model": r["llm_model"]
        }

# C4 after rerun
df["c4_rerun_verdict"] = df["c4_verdict"]
df["c4_rerun_source"] = df["c4_verdict_source"]
df["c4_rerun_confidence"] = df["c4_confidence"].astype(float)
df["c4_rerun_conflict"] = df["c4_conflict"].astype(bool)

for row_id, rdata in rerun_map.items():
    if row_id in df["id"].values:
        idx = df[df["id"] == row_id].index[0]
        df.loc[idx, "c4_rerun_verdict"] = rdata["verdict"]
        df.loc[idx, "c4_rerun_source"] = rdata["source"]
        if rdata["source"] == "llm_fallback":
            df.loc[idx, "c4_rerun_confidence"] = 0.85

n = len(df)
print(f"REPORT set N = {n}")

# 1. REPORT set C0..C5 metrics
classes = ["TRUE", "FALSE", "UNVERIFIABLE"]
configs = [
    ("C0", "c0_verdict"),
    ("C1", "c1_verdict"),
    ("C2", "c2_verdict"),
    ("C3", "c3_verdict"),
    ("C4 (as run)", "c4_verdict"),
    ("C4 (after rerun)", "c4_rerun_verdict"),
    ("C5", "c5_verdict"),
]

print("\n--- TABLE 1: REPORT SET (N=37) ---")
table1_rows = []
for name, col in configs:
    corr = (df[col] == df["gold"]).sum()
    p, lo, hi = wilson_ci(corr, n)
    macro_f1 = f1_score(df["gold"], df[col], labels=classes, average="macro", zero_division=0)
    unv = (df[col] == "UNVERIFIABLE").sum()
    unv_pct = unv / n * 100
    print(f"{name:<20} | Acc: {corr}/{n} ({p*100:.1f}%) [{lo*100:.1f}%, {hi*100:.1f}%] | Macro-F1: {macro_f1:.4f} | % UNV: {unv_pct:.1f}% ({unv}/{n})")
    table1_rows.append((name, f"{corr}/{n} ({p*100:.1f}%) [{lo*100:.1f}%, {hi*100:.1f}%]", f"{macro_f1:.4f}", f"{unv_pct:.1f}% ({unv}/{n})"))

# 2. Paired comparisons with rerun-corrected C4
print("\n--- TABLE 2: PAIRED COMPARISONS (C4 rerun vs Others) ---")
c4_corr = df["c4_rerun_verdict"] == df["gold"]

for other_name, other_col in [("C0", "c0_verdict"), ("C1", "c1_verdict"), ("C5", "c5_verdict")]:
    other_corr = df[other_col] == df["gold"]
    only_c4 = (c4_corr & ~other_corr).sum()
    only_other = (~c4_corr & other_corr).sum()
    both_right = (c4_corr & other_corr).sum()
    both_wrong = (~c4_corr & ~other_corr).sum()
    
    # Exact two-sided sign test on discordant pairs
    n_d = only_c4 + only_other
    if n_d > 0:
        res = binomtest(min(only_c4, only_other), n_d, 0.5, alternative="two-sided")
        pval = res.pvalue
    else:
        pval = 1.0
    print(f"C4 (rerun) vs {other_name:<2}: Only C4={only_c4}, Only {other_name}={only_other}, Both Right={both_right}, Both Wrong={both_wrong}, p={pval:.4f}")

# 3. Accuracy by category and language group
print("\n--- TABLE 3A: ACCURACY BY CATEGORY ---")
for cat, grp in df.groupby("category"):
    k = len(grp)
    c0_c = (grp["c0_verdict"] == grp["gold"]).sum()
    c4_c = (grp["c4_rerun_verdict"] == grp["gold"]).sum()
    c5_c = (grp["c5_verdict"] == grp["gold"]).sum()
    print(f"{cat:<20} (n={k:<2}) | C0: {c0_c}/{k} ({c0_c/k*100:.1f}%) | C4: {c4_c}/{k} ({c4_c/k*100:.1f}%) | C5: {c5_c}/{k} ({c5_c/k*100:.1f}%)")

print("\n--- TABLE 3B: ACCURACY BY LANGUAGE GROUP ---")
for lang_grp, grp in [("en", df[df["language"] == "en"]), ("non-en", df[df["language"] != "en"])]:
    k = len(grp)
    c0_c = (grp["c0_verdict"] == grp["gold"]).sum()
    c4_c = (grp["c4_rerun_verdict"] == grp["gold"]).sum()
    c5_c = (grp["c5_verdict"] == grp["gold"]).sum()
    print(f"{lang_grp:<20} (n={k:<2}) | C0: {c0_c}/{k} ({c0_c/k*100:.1f}%) | C4: {c4_c}/{k} ({c4_c/k*100:.1f}%) | C5: {c5_c}/{k} ({c5_c/k*100:.1f}%)")

# 4. C4 (after rerun) by evidence-strength band
print("\n--- TABLE 4: EVIDENCE STRENGTH BANDS (C4 after rerun) ---")
# Bands: confidence < 0.30 "weak", 0.30 <= conf <= 0.60 "moderate", conf > 0.60 "strong"
bands = [
    ("Weak (< 0.30)", df["c4_rerun_confidence"] < 0.30),
    ("Moderate (0.30 - 0.60)", (df["c4_rerun_confidence"] >= 0.30) & (df["c4_rerun_confidence"] <= 0.60)),
    ("Strong (> 0.60)", df["c4_rerun_confidence"] > 0.60),
]

for bname, mask in bands:
    bdf = df[mask]
    bn = len(bdf)
    b_corr = (bdf["c4_rerun_verdict"] == bdf["gold"]).sum()
    b_acc = b_corr / bn if bn > 0 else 0.0
    wrong_decisive = ((bdf["c4_rerun_verdict"].isin(["TRUE", "FALSE"])) & (bdf["c4_rerun_verdict"] != bdf["gold"])).sum()
    print(f"{bname:<25} | n={bn:<2} | Acc: {b_corr}/{bn} ({b_acc*100:.1f}%) | Wrong Decisive: {wrong_decisive}")

# 5. verifier_relation_pattern.csv 5 abstentions
print("\n--- TABLE 5: RELATION PATTERN ABSTENTIONS (N=5) ---")
df_rel = pd.read_csv("results/verifier_relation_pattern.csv")
rel_unv = df_rel[df_rel["c4_verdict"].apply(clean_v) == "UNVERIFIABLE"]
for _, r in rel_unv.iterrows():
    print(f"id={r['id']}, gold={r['gold']}, C4={r['c4_verdict']}, source={r['c4_verdict_source']}, claim='{r['claim_n']}'")

# 6. Remaining C4 errors after rerun
print("\n--- TABLE 6: REMAINING C4 ERRORS AFTER RERUN ---")
errors = df[df["c4_rerun_verdict"] != df["gold"]]
print(f"Total remaining errors: {len(errors)}")
for _, r in errors.iterrows():
    print(f"id={r['id']}, gold={r['gold']}, C4={r['c4_rerun_verdict']}, source={r['c4_rerun_source']}, conf={r['c4_rerun_confidence']:.2f}, conflict={r['c4_rerun_conflict']}, claim='{r['claim_n']}'")
