import sys
sys.path.insert(0, "src")

from retrieval.evidence_retriever import search_web, _simplify_query, retrieve_evidence

test_claim = "Turkey's capital is Ankara."
simplified = _simplify_query(test_claim)
print(f"Original: '{test_claim}' -> Simplified: '{simplified}'")
assert simplified == "Turkey capital is Ankara", f"Unexpected simplified query: {simplified}"

# Check another pattern claim
q2 = "Brazil's capital is Rio de Janeiro."
s2 = _simplify_query(q2)
print(f"Original: '{q2}' -> Simplified: '{s2}'")
assert s2 == "Brazil capital is Rio de Janeiro"

print("V9c query simplification test passed!")
