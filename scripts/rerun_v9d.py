import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import time
import math
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "src")

print("GROQ_API_KEY set:", bool(os.environ.get("GROQ_API_KEY")))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from retrieval.evidence_retriever import retrieve_evidence
from fact_check.fact_checker import (
    fact_check,
    evaluate_evidence,
    aggregate_results,
    build_alternative_query,
    get_cached_models
)

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

print("Loading cached models...")
semantic_model, tokenizer, nli_model = get_cached_models()

df_pattern = pd.read_csv("results/verifier_relation_pattern.csv")
df_pattern["gold"] = df_pattern["gold"].apply(clean_v)
df_pattern["c2_verdict"] = df_pattern["c2_verdict"].apply(clean_v)
df_pattern["c3_verdict"] = df_pattern["c3_verdict"].apply(clean_v)
df_pattern["c4_verdict"] = df_pattern["c4_verdict"].apply(clean_v)

# Zero evidence rows in verifier_relation_pattern.csv: R03p, R04p, R05p, R06p, R07p
target_ids = ["R03p", "R04p", "R05p", "R06p", "R07p"]
df_targets = df_pattern[df_pattern["id"].isin(target_ids)].copy()

print(f"Target rows to rerun: {len(df_targets)}")

rerun_records = []

for _, r in df_targets.iterrows():
    row_id = r["id"]
    gold = r["gold"]
    claim_n = r["claim_n"]
    print(f"\nEvaluating {row_id}: {claim_n} (Gold: {gold})")

    # Primary retrieval
    evidence = retrieve_evidence(claim_n, max_results=5)
    print(f"  Retrieved {len(evidence)} evidence passages")

    primary_results = evaluate_evidence(
        claim_n, evidence, tokenizer, nli_model, semantic_model
    ) if evidence else []

    # C2: NLI + consistency
    if primary_results:
        c2_agg = aggregate_results(primary_results)
        c2_v = clean_v(c2_agg[0])
        c2_conf = float(c2_agg[1])
    else:
        c2_v, c2_conf = "UNVERIFIABLE", 0.0

    # C3: C2 + competing-fact
    c3_v, c3_conf = c2_v, c2_conf
    if c2_v == "UNVERIFIABLE" and primary_results:
        claim_relation = primary_results[0].get("claim_relation")
        alt_query = build_alternative_query(claim_relation)
        if alt_query:
            print(f"  C3 competing query: {alt_query}")
            alt_evidence = retrieve_evidence(alt_query, max_results=5)
            seen_urls = {res["url"] for res in primary_results}
            alt_evidence = [e for e in alt_evidence if e.get("url") not in seen_urls]
            if alt_evidence:
                alt_results = evaluate_evidence(
                    claim_n, alt_evidence, tokenizer, nli_model, semantic_model
                )
                merged = primary_results + alt_results
                c3_agg = aggregate_results(merged)
                c3_v = clean_v(c3_agg[0])
                c3_conf = float(c3_agg[1])

    # C4: Full system
    c4_fc = fact_check(claim_n)
    if c4_fc:
        c4_v = clean_v(c4_fc.get("verdict"))
        c4_conf = float(c4_fc.get("confidence", 0.0))
        c4_src = c4_fc.get("verdict_source", "rule_based")
        c4_conflict = bool(c4_fc.get("conflict", False))
        c4_ev_count = len(c4_fc.get("evidence", []))
    else:
        c4_v, c4_conf, c4_src, c4_conflict, c4_ev_count = "UNVERIFIABLE", 0.0, "rule_based", False, 0

    print(f"  Results: C2={c2_v} ({c2_conf:.2f}), C3={c3_v} ({c3_conf:.2f}), C4={c4_v} ({c4_conf:.2f}, src={c4_src})")

    rerun_records.append({
        "id": row_id,
        "gold": gold,
        "claim_n": claim_n,
        "before_c2": r["c2_verdict"],
        "after_c2": c2_v,
        "before_c3": r["c3_verdict"],
        "after_c3": c3_v,
        "before_c4": r["c4_verdict"],
        "after_c4": c4_v,
        "after_c4_source": c4_src,
        "after_c4_confidence": c4_conf,
        "after_c4_conflict": c4_conflict,
        "after_c4_evidence_count": c4_ev_count,
    })
    time.sleep(2.0)

df_rerun = pd.DataFrame(rerun_records)
df_rerun.to_csv("results/verifier_relation_pattern_rerun.csv", index=False)

print("\n" + "="*80)
print("PER-ROW BEFORE/AFTER COMPARISON")
print("="*80)
for _, r in df_rerun.iterrows():
    print(f"{r['id']} (Gold: {r['gold']}):")
    print(f"  C2: {r['before_c2']} -> {r['after_c2']}")
    print(f"  C3: {r['before_c3']} -> {r['after_c3']}")
    print(f"  C4: {r['before_c4']} -> {r['after_c4']} (src={r['after_c4_source']}, ev={r['after_c4_evidence_count']})")

# Overall accuracy before/after on all 12 rows
n_total = len(df_pattern)
print("\n" + "="*80)
print(f"OVERALL ACCURACY ON RELATION PATTERN SET (N={n_total})")
print("="*80)

df_updated = df_pattern.copy()
for _, r in df_rerun.iterrows():
    idx = df_updated[df_updated["id"] == r["id"]].index[0]
    df_updated.loc[idx, "c2_verdict"] = r["after_c2"]
    df_updated.loc[idx, "c3_verdict"] = r["after_c3"]
    df_updated.loc[idx, "c4_verdict"] = r["after_c4"]

for c in ["c2", "c3", "c4"]:
    v_col = f"{c}_verdict"
    b_corr = (df_pattern[v_col] == df_pattern["gold"]).sum()
    bp, blo, bhi = wilson_ci(b_corr, n_total)
    
    a_corr = (df_updated[v_col] == df_updated["gold"]).sum()
    ap, alo, ahi = wilson_ci(a_corr, n_total)
    
    print(f"{c.upper()}:")
    print(f"  Before: {b_corr}/{n_total} ({bp*100:.1f}%) [Wilson 95% CI: {blo*100:.1f}%, {bhi*100:.1f}%]")
    print(f"  After:  {a_corr}/{n_total} ({ap*100:.1f}%) [Wilson 95% CI: {alo*100:.1f}%, {ahi*100:.1f}%]")
