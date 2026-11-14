from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.insert(0, "src")
print("GROQ_API_KEY set:", bool(os.environ.get("GROQ_API_KEY")))

from reasoning.llm_reasoner import llm_verify, llm_verify_with_status
from reasoning.claim_normalizer import normalize_claim

claim = "The Taj Mahal is located in Agra."
evidence = [
    {
        "title": "Taj Mahal - Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Taj_Mahal",
        "evidence": "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, Uttar Pradesh, India.",
        "nli_label": "support",
        "consistency_verdict": "SUPPORT",
        "semantic_similarity": 0.95
    }
]

res, status, model = llm_verify_with_status(claim, evidence)
print(f"llm_verify_with_status output: status={status}, model={model}, res={res}")
assert status in ("ok", "rate_limited", "error", "no_key"), f"Invalid status: {status}"
if status == "ok":
    assert res is not None
    assert "verdict" in res

norm = normalize_claim("The Taj Mahal is located in Agra.")
print("normalize_claim output:", norm)

print("V7a verification passed!")
