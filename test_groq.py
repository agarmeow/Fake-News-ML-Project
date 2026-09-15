import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
)

from reasoning.llm_reasoner import llm_verify

# Fake evidence in the same shape fact_checker.py produces -- just enough
# fields for llm_verify to build its prompt from.
fake_evidence = [
    {
        "title": "New Delhi | Britannica",
        "url": "https://www.britannica.com/place/New-Delhi",
        "nli_label": "entailment",
        "consistency_verdict": "SUPPORT",
        "semantic_similarity": 0.97,
        "evidence": "New Delhi is the national capital of India.",
    }
]

print("Testing Groq connection...")
print(f"GROQ_API_KEY set: {bool(os.environ.get('GROQ_API_KEY'))}")

result = llm_verify("India's capital city is New Delhi.", fake_evidence)

print("\nResult:")
print(result)

if result is None:
    print(
        "\nGot None -- either GROQ_API_KEY isn't set in this terminal, "
        "or the call/parse failed. Check the key and try again."
    )
else:
    print("\nGroq call succeeded.")