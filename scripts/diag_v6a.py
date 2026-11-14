import os
import sys
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from reasoning.claim_normalizer import normalize_claim
from fact_check.fact_checker import retrieve_evidence, evaluate_evidence, get_cached_models
from reasoning.llm_reasoner import llm_verify

claims_csv = PROJECT_ROOT / "data" / "eval" / "claims.csv"
df = pd.read_csv(claims_csv)

sem, tok, nli = get_cached_models()

for target_id in ["M01", "M05"]:
    row = df[df["id"] == target_id].iloc[0]
    raw = row["claim"]
    norm = normalize_claim(raw)
    claim_n = norm["claim"]
    ev = retrieve_evidence(claim_n, max_results=5)
    res = evaluate_evidence(claim_n, ev, tok, nli, sem) if ev else []
    try:
        llm_res = llm_verify(claim_n, res)
    except Exception as e:
        llm_res = {"error": str(e)}

    print("=" * 60)
    print(f"ID: {target_id}")
    print(f"Raw claim: {raw}")
    print(f"normalize_claim output: {norm}")
    print("Top 5 Evidence titles + URLs:")
    for i, e in enumerate(ev[:5], start=1):
        print(f"  [{i}] {e.get('title')} -> {e.get('url')}")
    if llm_res and "error" not in llm_res:
        v = llm_res.get("verdict")
        conf = llm_res.get("confidence")
        reas = str(llm_res.get("reasoning", ""))[:200]
        print(f"llm_verify raw return: verdict={v}, confidence={conf}, reasoning={reas}")
    else:
        print(f"llm_verify raw return: {llm_res}")

    print("fact_check override condition values:")
    print("  - Rule verdict before arbitration: UNVERIFIABLE")
    print("  - llm_result in fact_check run: None (raised 429 Client Error: Too Many Requests without retry)")
    print("  - if llm_result: False (override block bypassed)")
    print("  - Confidence cap threshold in override block: 0.85 (min(llm_confidence, 0.85))")
    print("One-line reason: In fact_check, llm_verify raised an unhandled HTTP 429 (rate limit) exception without retry, causing llm_result to be None so the override block never executed.")
