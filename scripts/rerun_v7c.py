from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, "src")
print("GROQ_API_KEY set:", bool(os.environ.get("GROQ_API_KEY")))
if not os.environ.get("GROQ_API_KEY"):
    print("ABORT: GROQ_API_KEY is not set.")
    sys.exit(1)

from fact_check.fact_checker import fact_check
from reasoning.llm_reasoner import GROQ_STATS

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

# Prepare targets
targets = []

# 1. verifier_ablation.csv
df_abl = pd.read_csv("results/verifier_ablation.csv")
exclude = [f"T{i:02d}" for i in range(1, 7)] + [f"F{i:02d}" for i in range(1, 7)]
df_abl_rep = df_abl[~df_abl["id"].isin(exclude)].copy()
df_abl_rep["gold"] = df_abl_rep["gold"].apply(clean_v)
df_abl_rep["c4_verdict"] = df_abl_rep["c4_verdict"].apply(clean_v)

for _, r in df_abl_rep.iterrows():
    if r["c4_verdict"] == "UNVERIFIABLE" and r["c4_verdict_source"] == "rule_based":
        targets.append({
            "file": "verifier_ablation.csv",
            "id": r["id"],
            "gold": r["gold"],
            "claim_n": r["claim_n"],
            "before_verdict": r["c4_verdict"],
        })

# 2. verifier_relation_pattern.csv
df_rel = pd.read_csv("results/verifier_relation_pattern.csv")
df_rel["gold"] = df_rel["gold"].apply(clean_v)
df_rel["c4_verdict"] = df_rel["c4_verdict"].apply(clean_v)

for _, r in df_rel.iterrows():
    if r["c4_verdict"] == "UNVERIFIABLE" and r["c4_verdict_source"] == "rule_based":
        targets.append({
            "file": "verifier_relation_pattern.csv",
            "id": r["id"],
            "gold": r["gold"],
            "claim_n": r["claim_n"],
            "before_verdict": r["c4_verdict"],
        })

print(f"Total rows to rerun: {len(targets)}")

rerun_csv = "results/verifier_rerun.csv"
done_map = {}
if os.path.exists(rerun_csv):
    df_existing = pd.read_csv(rerun_csv)
    for _, r in df_existing.iterrows():
        done_map[(r["file"], r["id"])] = r.to_dict()
else:
    with open(rerun_csv, "w", encoding="utf-8") as f:
        f.write("file,id,gold,before_verdict,after_verdict,after_source,llm_status,llm_model\n")

results_list = []
consecutive_groq_failures = 0

for item in targets:
    key = (item["file"], item["id"])
    if key in done_map:
        print(f"Skipping already run {key}: {done_map[key]['after_verdict']}")
        results_list.append(done_map[key])
        continue

    print(f"\nProcessing {item['file']} {item['id']}: {item['claim_n']}")
    res = fact_check(item["claim_n"])
    if res:
        after_v = res["verdict"]
        after_src = res["verdict_source"]
        llm_status = res.get("llm_status", "unknown")
        llm_model = res.get("llm_model", "none")
    else:
        after_v = "UNVERIFIABLE"
        after_src = "rule_based"
        llm_status = "ok"
        llm_model = "none"

    if llm_status != "ok":
        consecutive_groq_failures += 1
        if consecutive_groq_failures >= 5:
            print("GUARD: 5 consecutive Groq failures reached. Aborting.")
            sys.exit(1)
    else:
        consecutive_groq_failures = 0

    row_data = {
        "file": item["file"],
        "id": item["id"],
        "gold": item["gold"],
        "before_verdict": item["before_verdict"],
        "after_verdict": after_v,
        "after_source": after_src,
        "llm_status": llm_status,
        "llm_model": llm_model,
    }
    results_list.append(row_data)

    # Append to CSV
    with open(rerun_csv, "a", encoding="utf-8") as f:
        f.write(f"{item['file']},{item['id']},{item['gold']},{item['before_verdict']},{after_v},{after_src},{llm_status},{llm_model}\n")

    time.sleep(2.0)  # Pacing 2s

df_res = pd.DataFrame(results_list)
print("\n" + "="*80)
print("RERUN SUMMARY")
print("="*80)

# Evaluate per file
for fname, df_full, full_name in [("verifier_ablation.csv", df_abl_rep, "REPORT set (N=37)"),
                                   ("verifier_relation_pattern.csv", df_rel, "Pattern set (N=12)")]:
    sub_res = df_res[df_res["file"] == fname]
    df_merged = df_full.copy()
    
    # Original C4
    n = len(df_merged)
    orig_corr = (df_merged["c4_verdict"] == df_merged["gold"]).sum()
    orig_unv = (df_merged["c4_verdict"] == "UNVERIFIABLE").sum()
    p_orig, lo_orig, hi_orig = wilson_ci(orig_corr, n)

    # Updated C4
    after_dict = dict(zip(sub_res["id"], sub_res["after_verdict"].apply(clean_v)))
    df_merged["c4_after"] = df_merged["id"].map(after_dict).fillna(df_merged["c4_verdict"])
    
    new_corr = (df_merged["c4_after"] == df_merged["gold"]).sum()
    new_unv = (df_merged["c4_after"] == "UNVERIFIABLE").sum()
    p_new, lo_new, hi_new = wilson_ci(new_corr, n)

    changed = sub_res[sub_res["before_verdict"] != sub_res["after_verdict"]]
    not_ok = (sub_res["llm_status"] != "ok").sum()

    print(f"\n--- File: {fname} [{full_name}] ---")
    print(f"C4 as run:      Acc = {orig_corr}/{n} ({p_orig*100:.1f}%) [Wilson 95% CI: {lo_orig*100:.1f}%, {hi_orig*100:.1f}%], Abstentions = {orig_unv}/{n} ({orig_unv/n*100:.1f}%)")
    print(f"C4 after rerun:  Acc = {new_corr}/{n} ({p_new*100:.1f}%) [Wilson 95% CI: {lo_new*100:.1f}%, {hi_new*100:.1f}%], Abstentions = {new_unv}/{n} ({new_unv/n*100:.1f}%)")
    print(f"Rows with llm_status != ok after retries: {not_ok}")
    print(f"Rows changed ({len(changed)}):")
    for _, r in changed.iterrows():
        print(f"  id={r['id']}, gold={r['gold']}, before={r['before_verdict']}, after={r['after_verdict']} (source={r['after_source']}, status={r['llm_status']})")

print(f"\nTotal Groq calls attempted: {GROQ_STATS['attempted']}, failed: {GROQ_STATS['failed']}")
