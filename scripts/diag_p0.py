import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "src")

from fact_check.fact_checker import fact_check

claim = "Brazil's capital is Rio de Janeiro."
res = fact_check(claim)

print(f"R07p Fact Check Result:")
print(f"Verdict: {res['verdict']}, Confidence: {res['confidence']:.4f}, Conflict: {res.get('conflict', False)}, LLM Verdict: {res.get('llm_verdict', 'none')}")

print("\nCACHED EVIDENCE ROWS:")
header = f"{'Title[:40]':<40} | {'NLI Label':<12} | {'NLI Conf':<8} | {'Consistency':<12} | {'Claim Rel':<18} | {'Evidence Rel':<18}"
print(header)
print("-" * len(header))

for item in res.get("evidence", []):
    title = str(item.get("title", ""))[:40]
    nli_label = str(item.get("nli_label", "n/a"))
    nli_conf = f"{float(item.get('nli_confidence', 0.0)):.4f}"
    cons = str(item.get("consistency_verdict", "n/a"))
    cr = str(item.get("claim_relation", "None"))
    er = str(item.get("evidence_relation", "None"))
    print(f"{title:<40} | {nli_label:<12} | {nli_conf:<8} | {cons:<12} | {cr[:18]:<18} | {er[:18]:<18}")

print("\nPassage snippets:")
for i, item in enumerate(res.get("evidence", [])[:4], 1):
    print(f"[{i}] {item.get('title')}: {item.get('evidence')[:250]}...")
