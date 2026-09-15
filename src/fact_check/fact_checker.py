import sys
import os

# Allow imports from src/
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from retrieval.evidence_retriever import retrieve_evidence
from retrieval.semantic_ranker import semantic_rank_evidence

from reasoning.nli_reasoner import (
    load_nli_model,
    classify_claim_evidence
)

from verification.consistency_checker import (
    check_consistency
)

from sentence_transformers import SentenceTransformer


SEMANTIC_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

QUALITY_MAP = {
    "high": 1.0,
    "medium": 0.7,
    "unknown": 0.5
}

RELEVANCE_THRESHOLD = 0.35


def evaluate_evidence(claim, evidence_list, tokenizer, nli_model, semantic_model):
    """
    Rank a list of evidence items against the claim, then run
    NLI + entity/relation consistency checking on each of the
    top items. Returns a list of per-evidence-item result dicts.

    This is the same logic that used to live inline in fact_check(),
    pulled out so we can call it twice: once for evidence about the
    claim as stated, and again for evidence about the competing fact.
    """

    if not evidence_list:
        return []

    ranked_evidence = semantic_rank_evidence(
        claim,
        evidence_list,
        semantic_model
    )

    if not ranked_evidence:
        return []

    top_evidence = ranked_evidence[:5]

    results = []

    for item in top_evidence:

        nli_result = classify_claim_evidence(
            claim,
            item["text"],
            tokenizer,
            nli_model
        )

        consistency_result = check_consistency(
            claim,
            item["text"]
        )

        results.append({
            "title": item["title"],
            "url": item["url"],
            "source_quality": item["source_quality"],
            "semantic_similarity": item["semantic_similarity"],
            "combined_score": item["combined_score"],
            "nli_label": nli_result["label"],
            "nli_confidence": nli_result["confidence"],
            "consistency_verdict": consistency_result["verdict"],
            "entity_consistency": consistency_result["entity_consistency"],
            "relation_consistency": consistency_result["relation_consistency"],
            "consistency_score": consistency_result["consistency_score"],
            "claim_relation": consistency_result["claim_relation"],
            "evidence_relation": consistency_result["evidence_relation"],
            "evidence": item["text"],
        })

    return results


def aggregate_results(results):
    """
    Combine per-evidence results into a final verdict.

    Same decision rule as before:
      - SUPPORT needs NLI entailment AND consistency == SUPPORT
      - CONTRADICT needs NLI contradiction AND consistency == CONTRADICT
      - everything else is non-decisive
      - no decisive evidence at all -> UNVERIFIABLE
    """

    support_score = 0.0
    contradiction_score = 0.0

    supporting_evidence = []
    contradicting_evidence = []

    for result in results:

        semantic_similarity = max(result["semantic_similarity"], 0.0)

        sq_val = result.get("source_quality", 0.5)
        if isinstance(sq_val, str):
            source_quality = QUALITY_MAP.get(sq_val.lower(), 0.5)
        else:
            try:
                source_quality = float(sq_val)
            except (TypeError, ValueError):
                source_quality = 0.5

        source_quality = max(source_quality, 0.0)

        nli_confidence = max(result["nli_confidence"], 0.0)
        nli_label = result["nli_label"].lower()
        consistency_verdict = result["consistency_verdict"]

        if semantic_similarity < RELEVANCE_THRESHOLD:
            continue

        evidence_weight = (
            semantic_similarity
            * source_quality
            * nli_confidence
        )

        # ------------------------------------------------
        # NEUTRAL means the relation extractor found nothing
        # to check (the claim's predicate -- e.g. "died",
        # "announced" -- isn't in its known relation vocabulary).
        # That's an abstention, not a red flag, so we defer to
        # NLI in that case rather than silently discarding
        # otherwise-strong entailment/contradiction evidence.
        #
        # NOT_SUPPORT means it DID extract relations on both
        # sides and they disagree (e.g. the Mumbai/Maharashtra
        # case) -- that's an active structural signal against
        # trusting NLI, and still blocks scoring, same as before.
        # ------------------------------------------------

        consistency_abstained = result["claim_relation"] is None

        if "entail" in nli_label and (
            consistency_verdict == "SUPPORT"
            or (consistency_verdict == "NEUTRAL" and consistency_abstained)
        ):
            support_score += evidence_weight
            supporting_evidence.append(result)

        elif "contradict" in nli_label and (
            consistency_verdict == "CONTRADICT"
            or (consistency_verdict == "NEUTRAL" and consistency_abstained)
        ):
            contradiction_score += evidence_weight
            contradicting_evidence.append(result)

    total_decisive_score = support_score + contradiction_score

    if total_decisive_score == 0:
        verdict = "UNVERIFIABLE"
        confidence = 0.0

    elif support_score > contradiction_score:
        verdict = "TRUE"
        confidence = support_score / total_decisive_score

    elif contradiction_score > support_score:
        verdict = "FALSE"
        confidence = contradiction_score / total_decisive_score

    else:
        verdict = "UNVERIFIABLE"
        confidence = 0.0

    return (
        verdict,
        confidence,
        support_score,
        contradiction_score,
        supporting_evidence,
        contradicting_evidence,
    )


def build_alternative_query(claim_relation):
    """
    Given the claim's extracted (subject, relation, object) triple,
    build a search query for the COMPETING value of the same relation.

    Deliberately drops the object the claim asserted, so we are not
    hardcoding any specific competing fact (e.g. "New Delhi") — we let
    the search engine surface whatever the real value actually is.

    e.g. {"subject": "india", "relation": "capital", "object": "mumbai"}
         -> "india capital"
    """

    if not claim_relation:
        return None

    subject = claim_relation.get("subject")
    relation = claim_relation.get("relation")

    if not subject or not relation:
        return None

    return f"{subject} {relation}"


def fact_check(claim):

    print("\n" + "=" * 80)
    print("FACT-CHECKING CLAIM")
    print("=" * 80)
    print(f"\nClaim:\n{claim}")

    # --------------------------------------------------
    # STEP 1: RETRIEVE EVIDENCE (about the claim as stated)
    # --------------------------------------------------

    print("\n[1/6] Retrieving evidence...")

    evidence = retrieve_evidence(claim, max_results=5)

    if not evidence:
        print("No evidence found.")
        return

    print(f"Retrieved {len(evidence)} sources.")

    # --------------------------------------------------
    # STEP 2: LOAD MODELS (once, reused for both passes)
    # --------------------------------------------------

    print("\n[2/6] Loading models...")

    semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    tokenizer, nli_model = load_nli_model()

    # --------------------------------------------------
    # STEP 3: EVALUATE PRIMARY EVIDENCE
    # --------------------------------------------------

    print("\n[3/6] Verifying primary evidence...")

    results = evaluate_evidence(
        claim, evidence, tokenizer, nli_model, semantic_model
    )

    if not results:
        print("No usable evidence passages found.")
        return

    (
        verdict,
        confidence,
        support_score,
        contradiction_score,
        supporting_evidence,
        contradicting_evidence,
    ) = aggregate_results(results)

    # --------------------------------------------------
    # STEP 4: IF UNVERIFIABLE, GO LOOK FOR THE COMPETING FACT
    # --------------------------------------------------

    seen_urls = {r["url"] for r in results}

    if verdict == "UNVERIFIABLE":

        claim_relation = results[0]["claim_relation"]
        alt_query = build_alternative_query(claim_relation)

        if alt_query:

            print(
                f"\n[4/6] No decisive evidence found. "
                f"Searching for the competing fact: '{alt_query}'"
            )

            alt_evidence = retrieve_evidence(alt_query, max_results=5)

            # Don't re-process pages we already fetched in the primary pass
            alt_evidence = [
                e for e in alt_evidence if e["url"] not in seen_urls
            ]

            if alt_evidence:

                alt_results = evaluate_evidence(
                    claim, alt_evidence, tokenizer, nli_model, semantic_model
                )

                results = results + alt_results

                (
                    verdict,
                    confidence,
                    support_score,
                    contradiction_score,
                    supporting_evidence,
                    contradicting_evidence,
                ) = aggregate_results(results)

            else:
                print("[4/6] Alternative search returned nothing new.")

        else:
            print(
                "\n[4/6] No decisive evidence found, and no relation "
                "could be extracted from the claim to search for a "
                "competing fact — staying UNVERIFIABLE."
            )
    else:
        print("\n[4/6] Skipped competing-fact search (already decisive).")

    # --------------------------------------------------
    # STEP 5/6: DISPLAY FINAL RESULT
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL FACT-CHECK RESULT")
    print("=" * 80)

    print("\nClaim:")
    print(claim)

    print(f"\nVerdict: {verdict}")
    print(f"Confidence: {confidence:.4f}")

    print(f"\nSupporting evidence: {len(supporting_evidence)}")
    print(f"Contradicting evidence: {len(contradicting_evidence)}")

    print(f"\nSupport score: {support_score:.4f}")
    print(f"Contradiction score: {contradiction_score:.4f}")

    print("\n" + "-" * 80)
    print("EVIDENCE ANALYSIS")
    print("-" * 80)

    for i, result in enumerate(results, start=1):

        print(f"\nEvidence {i}")
        print(f"Title: {result['title']}")
        print(f"Source Quality: {result['source_quality']}")
        print(f"Semantic Similarity: {result['semantic_similarity']:.4f}")
        print(f"NLI: {result['nli_label']} ({result['nli_confidence']:.4f})")
        print(f"Consistency: {result['consistency_verdict']}")
        print(f"Entity Consistency: {result['entity_consistency']:.4f}")
        print(f"Relation Consistency: {result['relation_consistency']:.4f}")
        print(f"Consistency Score: {result['consistency_score']:.4f}")
        print(f"\nClaim Relation:\n{result['claim_relation']}")
        print(f"\nEvidence Relation:\n{result['evidence_relation']}")
        print(f"\nURL: {result['url']}")
        print(f"\nPassage:\n{result['evidence'][:600]}...")

    print("\n" + "=" * 80)

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "support_score": support_score,
        "contradiction_score": contradiction_score,
        "evidence": results,
    }


def main():

    claim = input("\nEnter claim to fact-check: ").strip()

    if not claim:
        print("No claim entered.")
        return

    fact_check(claim)


if __name__ == "__main__":
    main()