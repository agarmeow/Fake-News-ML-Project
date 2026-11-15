import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import time
import json
import math
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.stats import binomtest

load_dotenv()
sys.path.insert(0, "src")

api_key = os.environ.get("GROQ_API_KEY")
print("GROQ_API_KEY set:", bool(api_key))
if not api_key:
    print("ABORT: GROQ_API_KEY is not set.")
    sys.exit(1)

from reasoning.llm_reasoner import (
    llm_verify_with_status,
    _call_groq_chat,
    LAST_CALL_STATS,
    VALID_VERDICTS
)
from fact_check.fact_checker import (
    retrieve_evidence,
    evaluate_evidence,
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

def run_c0_call(claim_n, api_key):
    """
    Rerun closed-book call with the same retry/backoff wrapper and same prompt.
    Returns (verdict, confidence, status, model, attempts).
    """
    prompt = (
        "Using only your own knowledge, label the claim TRUE, FALSE or UNVERIFIABLE. "
        "If you are not sure, or it may concern events after your training data, answer UNVERIFIABLE. "
        "Respond strictly in valid JSON with keys 'verdict' and 'confidence' (float 0.0-1.0)."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Claim: {claim_n}"},
    ]

    primary_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    fallback_model = os.environ.get("GROQ_FAKETYPE_MODEL_2")

    backoffs = [2, 4, 8, 16]
    last_status = "error"
    active_model = primary_model

    for attempt in range(4):
        resp, code, retry_after, exc = _call_groq_chat(primary_model, api_key, messages, timeout=30)
        if resp is not None and code == 200:
            try:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                    if raw.lower().startswith("json"):
                        raw = raw.split("\n", 1)[-1]
                parsed = json.loads(raw)
                verdict = str(parsed.get("verdict", "")).upper().strip()
                if verdict in VALID_VERDICTS:
                    try:
                        conf = float(parsed.get("confidence", 0.5))
                    except (TypeError, ValueError):
                        conf = 0.5
                    return verdict, conf, "ok", primary_model, attempt + 1
                else:
                    return "UNVERIFIABLE", 0.0, "error", primary_model, attempt + 1
            except Exception:
                return "UNVERIFIABLE", 0.0, "error", primary_model, attempt + 1

        if code == 429:
            last_status = "rate_limited"
        elif code in (500, 502, 503, 408) or exc is not None:
            last_status = "error"
        else:
            last_status = "error"
            break

        if attempt < 3:
            sleep_sec = backoffs[attempt]
            if retry_after:
                try:
                    sleep_sec = float(retry_after)
                except (ValueError, TypeError):
                    pass
            sleep_sec = min(sleep_sec, 30)
            time.sleep(sleep_sec)

    # Fallback model attempt
    if fallback_model and fallback_model.strip():
        active_model = fallback_model.strip()
        time.sleep(2)
        resp, code, retry_after, exc = _call_groq_chat(active_model, api_key, messages, timeout=30)
        if resp is not None and code == 200:
            try:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                    if raw.lower().startswith("json"):
                        raw = raw.split("\n", 1)[-1]
                parsed = json.loads(raw)
                verdict = str(parsed.get("verdict", "")).upper().strip()
                if verdict in VALID_VERDICTS:
                    try:
                        conf = float(parsed.get("confidence", 0.5))
                    except (TypeError, ValueError):
                        conf = 0.5
                    return verdict, conf, "ok", active_model, 5
            except Exception:
                pass
        if code == 429:
            last_status = "rate_limited"
        else:
            last_status = "error"
        return "UNVERIFIABLE", 0.0, last_status, active_model, 5

    return "UNVERIFIABLE", 0.0, last_status, active_model, 4


# 1. Load data
df_abl = pd.read_csv("results/verifier_ablation.csv")
exclude = [f"T{i:02d}" for i in range(1, 7)] + [f"F{i:02d}" for i in range(1, 7)]
df = df_abl[~df_abl["id"].isin(exclude)].copy()
df["gold"] = df["gold"].apply(clean_v)
df["c0_verdict"] = df["c0_verdict"].apply(clean_v)
df["c5_verdict"] = df["c5_verdict"].apply(clean_v)

# Selection: C0 == UNVERIFIABLE, C5 == UNVERIFIABLE
c0_targets = df[df["c0_verdict"] == "UNVERIFIABLE"].copy()
c5_targets = df[df["c5_verdict"] == "UNVERIFIABLE"].copy()

print(f"REPORT set N = {len(df)}")
print(f"C0 targets (UNVERIFIABLE, N={len(c0_targets)}): {list(c0_targets['id'])}")
print(f"C5 targets (UNVERIFIABLE, N={len(c5_targets)}): {list(c5_targets['id'])}")

# Check resume
out_csv = "results/verifier_rerun_c0_c5.csv"
done_records = {}
if os.path.exists(out_csv):
    df_existing = pd.read_csv(out_csv)
    for _, r in df_existing.iterrows():
        done_records[(r["id"], r["config"])] = r.to_dict()
else:
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("id,gold,config,before,after,llm_status,llm_model,attempts\n")

print("\nLoading cached models for C5 evidence evaluation...")
semantic_model, tokenizer, nli_model = get_cached_models()

rerun_results = []
consecutive_failures = 0

# --- Process C0 targets ---
print("\n=== RUNNING C0 TARGETS ===")
for _, r in c0_targets.iterrows():
    row_id = r["id"]
    gold = r["gold"]
    claim_n = r["claim_n"]
    key = (row_id, "C0")

    if key in done_records:
        print(f"Skipping cached {key}: {done_records[key]['after']}")
        rerun_results.append(done_records[key])
        continue

    print(f"Evaluating C0 for {row_id}: '{claim_n[:60]}...'")
    v, conf, status, model, attempts = run_c0_call(claim_n, api_key)
    print(f"  -> C0 Result: {v} (conf={conf:.2f}, status={status}, model={model}, attempts={attempts})")

    if status != "ok":
        consecutive_failures += 1
        if consecutive_failures >= 5:
            print("GUARD: 5 consecutive Groq failures. Aborting.")
            sys.exit(1)
    else:
        consecutive_failures = 0

    rec = {
        "id": row_id,
        "gold": gold,
        "config": "C0",
        "before": "UNVERIFIABLE",
        "after": v,
        "llm_status": status,
        "llm_model": model,
        "attempts": attempts,
    }
    rerun_results.append(rec)
    with open(out_csv, "a", encoding="utf-8") as f:
        f.write(f"{row_id},{gold},C0,UNVERIFIABLE,{v},{status},{model},{attempts}\n")

    time.sleep(2.0)

# --- Process C5 targets ---
print("\n=== RUNNING C5 TARGETS ===")
for _, r in c5_targets.iterrows():
    row_id = r["id"]
    gold = r["gold"]
    claim_n = r["claim_n"]
    key = (row_id, "C5")

    if key in done_records:
        print(f"Skipping cached {key}: {done_records[key]['after']}")
        rerun_results.append(done_records[key])
        continue

    print(f"Evaluating C5 for {row_id}: '{claim_n[:60]}...'")
    # Get cached primary evidence
    evidence = retrieve_evidence(claim_n, max_results=5)
    primary_results = evaluate_evidence(claim_n, evidence, tokenizer, nli_model, semantic_model) if evidence else []

    res, status, model = llm_verify_with_status(claim_n, primary_results)
    attempts = LAST_CALL_STATS.get("attempts", 1)
    v = clean_v(res["verdict"]) if res else "UNVERIFIABLE"
    conf = res.get("confidence", 0.0) if res else 0.0
    print(f"  -> C5 Result: {v} (conf={conf:.2f}, status={status}, model={model}, attempts={attempts})")

    if status != "ok":
        consecutive_failures += 1
        if consecutive_failures >= 5:
            print("GUARD: 5 consecutive Groq failures. Aborting.")
            sys.exit(1)
    else:
        consecutive_failures = 0

    rec = {
        "id": row_id,
        "gold": gold,
        "config": "C5",
        "before": "UNVERIFIABLE",
        "after": v,
        "llm_status": status,
        "llm_model": model,
        "attempts": attempts,
    }
    rerun_results.append(rec)
    with open(out_csv, "a", encoding="utf-8") as f:
        f.write(f"{row_id},{gold},C5,UNVERIFIABLE,{v},{status},{model},{attempts}\n")

    time.sleep(2.0)

# ============================================================
# ANALYSIS & SUMMARY
# ============================================================
df_rr = pd.DataFrame(rerun_results)

# Create full updated columns for C0 and C5 on REPORT set
c0_map = dict(zip(df_rr[df_rr["config"] == "C0"]["id"], df_rr[df_rr["config"] == "C0"]["after"]))
c5_map = dict(zip(df_rr[df_rr["config"] == "C5"]["id"], df_rr[df_rr["config"] == "C5"]["after"]))

df["c0_after"] = df["id"].map(c0_map).fillna(df["c0_verdict"])
df["c5_after"] = df["id"].map(c5_map).fillna(df["c5_verdict"])

# Load C4 after rerun from results/verifier_rerun.csv
c4_rerun_csv = "results/verifier_rerun.csv"
c4_map = {}
if os.path.exists(c4_rerun_csv):
    df_c4_rr = pd.read_csv(c4_rerun_csv)
    for _, cr in df_c4_rr[df_c4_rr["file"] == "verifier_ablation.csv"].iterrows():
        c4_map[cr["id"]] = clean_v(cr["after_verdict"])
df["c4_after"] = df["id"].map(c4_map).fillna(df["c4_verdict"].apply(clean_v))

N = len(df)
print("\n" + "="*80)
print("SYMMETRIC RELIABILITY RERUN SUMMARY")
print("="*80)

# 1. C0 Before vs After
c0_b_corr = (df["c0_verdict"] == df["gold"]).sum()
c0_b_unv = (df["c0_verdict"] == "UNVERIFIABLE").sum()
p_c0_b, lo_c0_b, hi_c0_b = wilson_ci(c0_b_corr, N)

c0_a_corr = (df["c0_after"] == df["gold"]).sum()
c0_a_unv = (df["c0_after"] == "UNVERIFIABLE").sum()
p_c0_a, lo_c0_a, hi_c0_a = wilson_ci(c0_a_corr, N)

c0_changed = df_rr[(df_rr["config"] == "C0") & (df_rr["before"] != df_rr["after"])]
c0_not_ok = (df_rr[df_rr["config"] == "C0"]["llm_status"] != "ok").sum()

print(f"C0 Before: Acc = {c0_b_corr}/{N} ({p_c0_b*100:.1f}%) [{lo_c0_b*100:.1f}%, {hi_c0_b*100:.1f}%], Abstentions = {c0_b_unv}/{N}")
print(f"C0 After:  Acc = {c0_a_corr}/{N} ({p_c0_a*100:.1f}%) [{lo_c0_a*100:.1f}%, {hi_c0_a*100:.1f}%], Abstentions = {c0_a_unv}/{N}")
print(f"C0 Rows Changed ({len(c0_changed)}):")
for _, r in c0_changed.iterrows():
    print(f"  {r['id']} (Gold: {r['gold']}): {r['before']} -> {r['after']} (status={r['llm_status']})")
print(f"C0 status != ok: {c0_not_ok}")

# 2. C5 Before vs After
c5_b_corr = (df["c5_verdict"] == df["gold"]).sum()
c5_b_unv = (df["c5_verdict"] == "UNVERIFIABLE").sum()
p_c5_b, lo_c5_b, hi_c5_b = wilson_ci(c5_b_corr, N)

c5_a_corr = (df["c5_after"] == df["gold"]).sum()
c5_a_unv = (df["c5_after"] == "UNVERIFIABLE").sum()
p_c5_a, lo_c5_a, hi_c5_a = wilson_ci(c5_a_corr, N)

c5_changed = df_rr[(df_rr["config"] == "C5") & (df_rr["before"] != df_rr["after"])]
c5_not_ok = (df_rr[df_rr["config"] == "C5"]["llm_status"] != "ok").sum()

print(f"\nC5 Before: Acc = {c5_b_corr}/{N} ({p_c5_b*100:.1f}%) [{lo_c5_b*100:.1f}%, {hi_c5_b*100:.1f}%], Abstentions = {c5_b_unv}/{N}")
print(f"C5 After:  Acc = {c5_a_corr}/{N} ({p_c5_a*100:.1f}%) [{lo_c5_a*100:.1f}%, {hi_c5_a*100:.1f}%], Abstentions = {c5_a_unv}/{N}")
print(f"C5 Rows Changed ({len(c5_changed)}):")
for _, r in c5_changed.iterrows():
    print(f"  {r['id']} (Gold: {r['gold']}): {r['before']} -> {r['after']} (status={r['llm_status']})")
print(f"C5 status != ok: {c5_not_ok}")

# 3. Paired Comparisons: C4 (after) vs C5 (after), C4 (after) vs C0 (after)
print("\n" + "="*80)
print("PAIRED COMPARISONS (C4 after rerun vs Others)")
print("="*80)

c4_correct = df["c4_after"] == df["gold"]

for other_name, other_col in [("C5 (after)", "c5_after"), ("C0 (after)", "c0_after")]:
    o_corr = df[other_col] == df["gold"]
    only_c4 = (c4_correct & ~o_corr).sum()
    only_o = (~c4_correct & o_corr).sum()
    both_right = (c4_correct & o_corr).sum()
    both_wrong = (~c4_correct & ~o_corr).sum()
    n_d = only_c4 + only_o
    if n_d > 0:
        pval = binomtest(min(only_c4, only_o), n_d, 0.5, alternative="two-sided").pvalue
    else:
        pval = 1.0
    print(f"C4 (after) vs {other_name}:")
    print(f"  Only C4 Right: {only_c4} | Only {other_name} Right: {only_o} | Both Right: {both_right} | Both Wrong: {both_wrong} | Discordant: {n_d} | Exact sign-test p: {pval:.4f}")
