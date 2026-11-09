import sys
import os
import time
import math
import re
import difflib
from functools import lru_cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Allow imports from src/
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from retrieval.evidence_retriever import (
    retrieve_evidence,
    CACHE_STATS,
    TIMING_STATS as RETRIEVER_TIMINGS,
    reset_stats as reset_retriever_stats
)
from retrieval.semantic_ranker import semantic_rank_evidence

from reasoning.nli_reasoner import (
    load_nli_model,
    classify_claim_evidence,
    classify_claim_evidence_batch
)

from verification.consistency_checker import (
    check_consistency,
    RELATION_MAP,
    nlp
)
from reasoning.llm_reasoner import llm_verify

from sentence_transformers import SentenceTransformer


SEMANTIC_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

STAGE_TIMINGS = {
    "search_web": 0.0,
    "page_fetch": 0.0,
    "semantic_ranking": 0.0,
    "nli_loop": 0.0,
    "consistency_loop": 0.0,
    "llm_verify": 0.0,
    "model_loading": 0.0,
}

QUALITY_MAP = {
    "high": 1.0,
    "medium": 0.7,
    "unknown": 0.5
}

RELEVANCE_THRESHOLD = 0.35


@lru_cache(maxsize=1)
def get_cached_models():
    """Load and cache SentenceTransformer and NLI model once."""
    semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    tokenizer, nli_model = load_nli_model()
    return semantic_model, tokenizer, nli_model


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

    t_sem = time.time()
    ranked_evidence = semantic_rank_evidence(
        claim,
        evidence_list,
        semantic_model
    )
    STAGE_TIMINGS["semantic_ranking"] += (time.time() - t_sem)

    if not ranked_evidence:
        return []

    top_evidence = ranked_evidence[:5]

    t_nli = time.time()
    passages = [item["text"] for item in top_evidence]
    nli_results = classify_claim_evidence_batch(
        claim,
        passages,
        tokenizer,
        nli_model
    )
    STAGE_TIMINGS["nli_loop"] += (time.time() - t_nli)

    results = []

    for idx, item in enumerate(top_evidence):
        nli_result = nli_results[idx]

        t_cons = time.time()
        consistency_result = check_consistency(
            claim,
            item["text"]
        )
        STAGE_TIMINGS["consistency_loop"] += (time.time() - t_cons)

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
            "claim": claim,
        })

    return results


def get_primary_entity(claim):
    """
    Primary entity = first PERSON entity of the claim (spaCy),
    else the first named entity (GPE/ORG/LOC/EVENT/WORK_OF_ART/NORP).
    If no named entity, returns None.
    """
    if not claim:
        return None
    doc = nlp(claim)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent
    allowed_types = {"GPE", "ORG", "LOC", "EVENT", "WORK_OF_ART", "NORP"}
    for ent in doc.ents:
        if ent.label_ in allowed_types:
            return ent
    return None


def check_entity_in_passage(primary_entity, passage):
    """
    Check if ALL tokens of primary_entity appear in the passage.
    Case-insensitive, allows per-token fuzzy ratio >= 0.9 for OCR typos.
    A first-name-only match is not enough.
    """
    if primary_entity is None:
        return True
    if not passage:
        return False

    passage_lower = passage.lower()
    ent_tokens = [t.text.lower() for t in primary_entity if t.text.isalnum()]
    if not ent_tokens:
        return True

    passage_words = set(re.findall(r"\b[a-z0-9]+\b", passage_lower))

    for et in ent_tokens:
        if et in passage_words or re.search(r"\b" + re.escape(et) + r"\b", passage_lower):
            continue
        matched_fuzzy = False
        for pw in passage_words:
            if abs(len(pw) - len(et)) <= 2:
                if difflib.SequenceMatcher(None, et, pw).ratio() >= 0.9:
                    matched_fuzzy = True
                    break
        if not matched_fuzzy:
            return False

    return True


def aggregate_results(results):
    """
    Combine per-evidence results into a final verdict.
    Note: confidence is an evidence score, not a calibrated probability.

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

    claim_str = results[0]["claim"] if results and "claim" in results[0] else ""
    primary_entity = get_primary_entity(claim_str) if claim_str else None

    for result in results:
        result["counted_as"] = "none"

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

        # F6: Primary entity guard
        passage = result.get("evidence", "")
        entity_guard_ok = check_entity_in_passage(primary_entity, passage)
        result["entity_guard_ok"] = entity_guard_ok
        if not entity_guard_ok:
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

        # T4: Keyword guard - if claim_relation is None but claim contains a trigger word
        # of any existing relation pattern, treat as NOT abstained.
        if consistency_abstained:
            c_str = result.get("claim", "")
            if c_str:
                claim_tokens = {w.lower().strip(".,'\"?!;:()[]{}") for w in c_str.split()}
                trigger_words = set(RELATION_MAP.keys())
                if claim_tokens.intersection(trigger_words):
                    consistency_abstained = False

        if "entail" in nli_label and (
            consistency_verdict == "SUPPORT"
            or (consistency_verdict == "NEUTRAL" and consistency_abstained)
        ):
            support_score += evidence_weight
            supporting_evidence.append(result)
            result["counted_as"] = "support"

        elif "contradict" in nli_label and (
            consistency_verdict == "CONTRADICT"
            or (consistency_verdict == "NEUTRAL" and consistency_abstained)
        ):
            contradiction_score += evidence_weight
            contradicting_evidence.append(result)
            result["counted_as"] = "contradict"

    total_decisive_score = support_score + contradiction_score

    if total_decisive_score == 0:
        verdict = "UNVERIFIABLE"
        confidence = 0.0

    elif support_score > contradiction_score:
        verdict = "TRUE"
        dominant = support_score
        agreement = dominant / total_decisive_score
        strength = 1.0 - math.exp(-total_decisive_score)
        confidence = min(0.99, agreement * strength)

    elif contradiction_score > support_score:
        verdict = "FALSE"
        dominant = contradiction_score
        agreement = dominant / total_decisive_score
        strength = 1.0 - math.exp(-total_decisive_score)
        confidence = min(0.99, agreement * strength)

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

    for k in STAGE_TIMINGS:
        STAGE_TIMINGS[k] = 0.0
    reset_retriever_stats()

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

    t_load = time.time()
    semantic_model, tokenizer, nli_model = get_cached_models()
    STAGE_TIMINGS["model_loading"] = time.time() - t_load

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
    # STEP 4.5: LLM ARBITRATION (Groq gpt-oss) -- additive only.
    # Reads the SAME evidence already retrieved above (no new search).
    # Never overrides a decisive rule-based verdict -- only steps in
    # when the rule-based pipeline is UNVERIFIABLE. Falls back
    # silently to the rule-based verdict if Groq is unreachable, the
    # key is missing, or the response doesn't parse.
    # --------------------------------------------------

    llm_result = None
    llm_reasoning = None
    verdict_source = "rule_based"
    conflict = False
    llm_verdict = None

    t_llm = time.time()
    try:
        llm_result = llm_verify(claim, results)
    except Exception as e:
        print(f"\n[LLM] Groq call failed, falling back to rule-based verdict: {e}")
        llm_result = None
    STAGE_TIMINGS["llm_verify"] = time.time() - t_llm

    if llm_result:
        llm_reasoning = llm_result["reasoning"]
        llm_v = llm_result.get("verdict")
        llm_verdict = llm_v

        # T8: Conflict flag if rule-based verdict is TRUE/FALSE and llm returns opposite decisive verdict
        if verdict in {"TRUE", "FALSE"} and llm_v in {"TRUE", "FALSE"} and verdict != llm_v:
            conflict = True

        if verdict == "UNVERIFIABLE" and llm_v != "UNVERIFIABLE":
            verdict = llm_v
            confidence = min(llm_result["confidence"], 0.85)
            verdict_source = "llm_fallback"
            print(
                f"\n[LLM] Rule-based pipeline was UNVERIFIABLE; "
                f"adopting LLM verdict: {verdict} ({confidence:.2f})"
            )

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

    STAGE_TIMINGS["search_web"] = RETRIEVER_TIMINGS["search_web"]
    STAGE_TIMINGS["page_fetch"] = RETRIEVER_TIMINGS["page_fetch"]

    if os.environ.get("FC_TIMING") == "1":
        print("\n" + "=" * 50)
        print("PER-STAGE TIMINGS (FC_TIMING=1)")
        print("=" * 50)
        for stage, sec in STAGE_TIMINGS.items():
            print(f"  {stage:<20}: {sec:.4f}s")
        print(f"  {'cache_stats':<20}: {CACHE_STATS}")
        print("=" * 50)

    ret = {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "support_score": support_score,
        "contradiction_score": contradiction_score,
        "evidence": results,
        "llm_reasoning": llm_reasoning,
        "verdict_source": verdict_source,
        "timings": dict(STAGE_TIMINGS),
        "cache_stats": dict(CACHE_STATS),
    }
    if conflict:
        ret["conflict"] = True
        ret["llm_verdict"] = llm_verdict

    return ret


def main():

    claim = input("\nEnter claim to fact-check: ").strip()

    if not claim:
        print("No claim entered.")
        return

    fact_check(claim)


if __name__ == "__main__":
    main()