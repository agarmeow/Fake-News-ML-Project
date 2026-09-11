import re
import sys
from pathlib import Path

# Allow execution from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import spacy


# ============================================================
# CONFIGURATION
# ============================================================

SPACY_MODEL = "en_core_web_sm"


# Normalize different word forms into the same relation.
RELATION_MAP = {
    "capital": "capital",
    "capitals": "capital",

    "president": "president",
    "presidents": "president",

    "prime": "prime minister",

    "located": "located",
    "situated": "located",

    "founded": "founded",
    "established": "founded",

    "born": "born",
    "died": "died",

    "invented": "invented",
    "discovered": "discovered",

    "won": "won",
    "banned": "banned",
    "ban": "banned",
    "prohibited": "banned",

    "approved": "approved",
    "created": "created",
    "launched": "launched",
}


NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "without",

    "didn't",
    "doesn't",
    "don't",
    "isn't",
    "wasn't",
    "weren't",

    "cannot",
    "can't",
    "couldn't",

    "won't",
    "wouldn't",

    "hasn't",
    "haven't",
    "hadn't",
}


# ============================================================
# LOAD SPACY
# ============================================================

try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    raise OSError(
        f"spaCy model '{SPACY_MODEL}' is not installed.\n\n"
        f"Run:\n"
        f"python -m spacy download {SPACY_MODEL}"
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize whitespace."""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_entity(text):
    """Normalize an entity for comparison."""

    if text is None:
        return ""

    text = text.lower().strip()

    # Remove possessive ending.
    text = re.sub(r"'s$", "", text)

    # Remove surrounding punctuation.
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(text):
    """
    Extract named entities using spaCy.

    We do NOT treat every noun chunk as an entity here.
    This avoids noisy entities such as:

        "the capital"
        "the largest country"

    which caused misleading overlap in the previous version.
    """

    text = normalize_text(text)
    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        value = normalize_entity(ent.text)

        if value and value not in entities:
            entities.append(value)

    return entities


# ============================================================
# ENTITY TEXT FROM SPACY TOKEN
# ============================================================

def get_entity_text(token):
    """
    Obtain the meaningful noun phrase surrounding a token.
    Used mainly for subjects and objects.
    """
    if token is None:
        return ""

    # If spaCy recognized a named entity containing this token,
    # use the complete entity.
    for ent in token.doc.ents:
        if ent.start <= token.i < ent.end:
            return normalize_entity(ent.text)

    # Handle constructions like "the city of Mumbai", "state of Maharashtra"
    if token.text.lower() in {"city", "state", "town", "country", "province", "district"}:
        for child in token.children:
            if child.dep_ == "prep" and child.text.lower() == "of":
                for gc in child.children:
                    if gc.dep_ in {"pobj", "obj"}:
                        return get_entity_text(gc)

    # Otherwise use the token itself.
    return normalize_entity(token.text)


# ============================================================
# CAPITAL MODIFIER DEFINITIONS
# ============================================================

NON_ADMINISTRATIVE_CAPITAL_MODIFIERS = {
    "commercial",
    "financial",
    "economic",
    "cultural",
    "entertainment",
    "summer",
    "winter",
    "industrial",
    "business",
    "fashion",
    "tech",
    "technology",
    "culinary",
    "sports",
}

def get_capital_modifier(capital_token):
    """
    Check if 'capital' has a qualifying modifier like 'commercial', 'financial', etc.
    """
    for child in capital_token.children:
        if child.dep_ in {"amod", "compound"} and child.text.lower() not in {"the", "a", "an", "city"}:
            mod = child.lemma_.lower()
            if mod in NON_ADMINISTRATIVE_CAPITAL_MODIFIERS:
                return f"{mod}_capital"
    return "capital"


# ============================================================
# RELATION NORMALIZATION
# ============================================================

def normalize_relation(word):
    """Convert relation variants to a common relation name."""

    word = word.lower().strip()

    return RELATION_MAP.get(word)


# ============================================================
# NEGATION DETECTION
# ============================================================

def contains_negation(text):
    """
    Detect explicit negation.

    Uses both token-level checking and spaCy dependency
    information where possible.
    """

    text = normalize_text(text)

    doc = nlp(text)

    # First check spaCy's negation dependency.
    for token in doc:
        if token.dep_ == "neg":
            return True

    # Fallback for contractions / explicit forms.
    lowered = text.lower()

    for word in NEGATION_WORDS:
        if re.search(
            r"\b" + re.escape(word) + r"\b",
            lowered
        ):
            return True

    return False


# ============================================================
# CAPITAL RELATION EXTRACTION
# ============================================================

def extract_capital_relation(doc):
    """
    Extract capital relationships from copular sentences.

    Handles structures such as:

        India's capital city is New Delhi.
        New Delhi is the capital of India.
        Mumbai is India's capital.
        Mumbai is the capital of Maharashtra.

    All are normalized into:

        subject -> capital -> object

    where the subject is the country/state whose capital
    is being described.
    """

    for token in doc:

        # We are interested in forms of "be".
        if token.lemma_.lower() != "be":
            continue

        # ----------------------------------------------------
        # Find the grammatical subject.
        # ----------------------------------------------------

        subjects = [
            child
            for child in token.children
            if child.dep_ in {"nsubj", "nsubjpass"}
        ]

        if not subjects:
            continue

        grammatical_subject = subjects[0]

        # ----------------------------------------------------
        # Find the complement.
        # ----------------------------------------------------

        complements = [
            child
            for child in token.children
            if child.dep_ in {"attr", "acomp", "oprd"}
        ]

        # ====================================================
        # CASE A
        #
        # India's capital city is New Delhi.
        #
        # grammatical subject = capital city
        # possessive modifier = India
        # complement = New Delhi
        #
        # Desired:
        #
        # India -> capital -> New Delhi
        # ====================================================

        capital_token = None
        if grammatical_subject.lemma_.lower() == "capital":
            capital_token = grammatical_subject
        else:
            for child in grammatical_subject.children:
                if child.lemma_.lower() == "capital":
                    capital_token = child
                    break

        if capital_token is not None:
            subject_possessor = None
            for child in grammatical_subject.children:
                if child.dep_ == "poss":
                    subject_possessor = child
                    break

            if subject_possessor is not None and complements:
                possessor_text = get_entity_text(subject_possessor)
                object_text = get_entity_text(complements[0])
                if possessor_text and object_text:
                    rel = get_capital_modifier(capital_token)
                    return {
                        "relation": rel,
                        "subject": possessor_text,
                        "object": object_text,
                        "negated": contains_negation(doc.text),
                    }

        # ====================================================
        # CASE B
        #
        # Mumbai is India's capital.
        #
        # grammatical subject = Mumbai
        # complement = capital
        # possessive = India
        #
        # Desired:
        #
        # India -> capital -> Mumbai
        # ====================================================

        for complement in complements:

            if complement.lemma_.lower() != "capital":
                continue

            possessor = None

            for child in complement.children:
                if child.dep_ == "poss":
                    possessor = child
                    break

            if possessor is not None:

                subject_text = get_entity_text(
                    possessor
                )

                object_text = get_entity_text(
                    grammatical_subject
                )

                if subject_text and object_text:
                    rel = get_capital_modifier(complement)
                    return {
                        "relation": rel,
                        "subject": subject_text,
                        "object": object_text,
                        "negated": contains_negation(doc.text),
                    }

        # ====================================================
        # CASE C
        #
        # Mumbai is the capital of Maharashtra.
        #
        # grammatical subject = Mumbai
        # complement = capital
        # preposition "of" -> Maharashtra
        #
        # Desired:
        #
        # Maharashtra -> capital -> Mumbai
        # ====================================================

        for complement in complements:

            # Handle both:
            #   "the capital of Maharashtra"
            #   "the capital city of Maharashtra"
            #
            # In "capital city", spaCy makes "city" the
            # complement and "capital" its compound modifier.

            capital_token = complement

            if complement.lemma_.lower() == "city":
                capital_children = [
                    child
                    for child in complement.children
                    if child.lemma_.lower() == "capital"
                    and child.dep_ == "compound"
                ]

                if not capital_children:
                    continue

                capital_token = capital_children[0]

            elif complement.lemma_.lower() != "capital":
                continue

            for child in complement.children:

                if child.dep_ == "prep" and child.text.lower() == "of":

                    objects = [
                        grandchild
                        for grandchild in child.children
                        if grandchild.dep_ in {
                            "pobj",
                            "obj"
                        }
                    ]

                    if objects:

                        subject_text = get_entity_text(
                            objects[0]
                        )

                        object_text = get_entity_text(
                            grammatical_subject
                        )

                        if subject_text and object_text:
                            rel = get_capital_modifier(capital_token)
                            return {
                                "relation": rel,
                                "subject": subject_text,
                                "object": object_text,
                                "negated": contains_negation(
                                    doc.text
                                ),
                            }

    return None


# ============================================================
# VERB RELATION EXTRACTION
# ============================================================

def extract_verb_relation(doc):
    """
    Extract simple subject-verb-object relationships.

    Example:

        India banned X.

    becomes:

        India -> banned -> X
    """

    for token in doc:

        # We only inspect verbs.
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        relation = normalize_relation(token.lemma_)

        if relation is None:
            continue

        # Ignore "be" here because copular relations are
        # handled separately.
        if token.lemma_.lower() == "be":
            continue

        subject = None
        object_token = None

        # ----------------------------------------------------
        # Find subject
        # ----------------------------------------------------

        for child in token.children:

            if child.dep_ in {
                "nsubj",
                "nsubjpass"
            }:

                subject = child
                break

        # ----------------------------------------------------
        # Find direct object
        # ----------------------------------------------------

        for child in token.children:

            if child.dep_ in {
                "dobj",
                "obj",
                "attr"
            }:

                object_token = child
                break

        if subject is None or object_token is None:
            continue

        subject_text = get_entity_text(subject)
        object_text = get_entity_text(object_token)

        if not subject_text or not object_text:
            continue

        return {
            "relation": relation,
            "subject": subject_text,
            "object": object_text,
            "negated": contains_negation(doc.text),
        }

    return None


# ============================================================
# MAIN RELATION EXTRACTION
# ============================================================

def extract_relation(text):
    """
    Extract a normalized factual relationship.

    Priority:
        1. Capital/coplanar factual relation
        2. Simple verb relation
    """

    text = normalize_text(text)

    doc = nlp(text)

    relation = extract_capital_relation(doc)

    if relation is not None:
        return relation

    relation = extract_verb_relation(doc)

    if relation is not None:
        return relation

    return None


# ============================================================
# ENTITY CONSISTENCY
# ============================================================

def check_entity_consistency(
    claim_entities,
    evidence_entities
):
    """
    Compare meaningful named entities.

    Returns:
        1.0 = all claim entities are present
        partial score = some overlap
        0.0 = no overlap
    """

    if not claim_entities:
        return 0.0

    if not evidence_entities:
        return 0.0

    claim_set = set(claim_entities)
    evidence_set = set(evidence_entities)

    shared = claim_set.intersection(
        evidence_set
    )

    return round(
        len(shared) / len(claim_set),
        4
    )


# ============================================================
# RELATION COMPARISON
# ============================================================

def compare_relations(
    claim_relation,
    evidence_relation
):
    """
    Compare normalized factual relationships.

    Returns one of:

        SUPPORT
        CONTRADICT
        NOT_SUPPORT
        NEUTRAL
    """

    # If either sentence has no extractable relation,
    # consistency alone cannot establish a relationship.
    if (
        claim_relation is None
        or evidence_relation is None
    ):
        return "NEUTRAL"

    claim_rel = claim_relation["relation"]
    evidence_rel = evidence_relation["relation"]

    claim_subject = claim_relation["subject"]
    claim_object = claim_relation["object"]

    evidence_subject = evidence_relation["subject"]
    evidence_object = evidence_relation["object"]

    # --------------------------------------------------------
    # Exact same relationship
    # --------------------------------------------------------

    same_subject = (
        claim_subject == evidence_subject
    )

    same_object = (
        claim_object == evidence_object
    )

    # --------------------------------------------------------
    # Different relations
    # --------------------------------------------------------

    if claim_rel != evidence_rel:
        # If one is a qualified variant of capital (e.g. "commercial_capital")
        # and the other is the administrative capital, and subjects or objects align:
        if "capital" in claim_rel and "capital" in evidence_rel:
            if (same_subject and same_object) or (same_object and not same_subject):
                return "NOT_SUPPORT"
        return "NEUTRAL"

    # --------------------------------------------------------
    # Negation mismatch
    # --------------------------------------------------------

    claim_negated = claim_relation["negated"]
    evidence_negated = evidence_relation["negated"]

    negation_mismatch = (
        claim_negated != evidence_negated
    )

    if (
        same_subject
        and same_object
        and negation_mismatch
    ):
        return "CONTRADICT"

    # --------------------------------------------------------
    # Exact same relationship and polarity
    # --------------------------------------------------------

    if (
        same_subject
        and same_object
    ):
        return "SUPPORT"

    # --------------------------------------------------------
    # Same subject + same relation,
    # but different object.
    #
    # Example:
    #
    # Claim:
    # India -> capital -> Mumbai
    #
    # Evidence:
    # India -> capital -> New Delhi
    #
    # This is a direct contradiction.
    # --------------------------------------------------------

    if (
        same_subject
        and not same_object
    ):
        return "CONTRADICT"

    # --------------------------------------------------------
    # Same object + same relation,
    # but different subject.
    #
    # Example:
    #
    # Claim:
    # India -> capital -> Mumbai
    #
    # Evidence:
    # Maharashtra -> capital -> Mumbai
    #
    # This does NOT support the claim.
    # --------------------------------------------------------

    if (
        same_object
        and not same_subject
    ):
        return "NOT_SUPPORT"

    # --------------------------------------------------------
    # Same relation but neither side matches.
    # --------------------------------------------------------

    return "NEUTRAL"


# ============================================================
# COMPLETE CONSISTENCY CHECK
# ============================================================

def check_consistency(
    claim,
    evidence
):
    """
    Run the complete consistency analysis.
    """

    claim = normalize_text(claim)
    evidence = normalize_text(evidence)

    # --------------------------------------------------------
    # Extract entities
    # --------------------------------------------------------

    claim_entities = extract_entities(
        claim
    )

    evidence_entities = extract_entities(
        evidence
    )

    # --------------------------------------------------------
    # Extract relations
    # --------------------------------------------------------

    claim_relation = extract_relation(
        claim
    )

    evidence_relation = extract_relation(
        evidence
    )

    # --------------------------------------------------------
    # Entity score
    # --------------------------------------------------------

    entity_score = check_entity_consistency(
        claim_entities,
        evidence_entities
    )

    # --------------------------------------------------------
    # Relationship verdict
    # --------------------------------------------------------

    verdict = compare_relations(
        claim_relation,
        evidence_relation
    )

    # --------------------------------------------------------
    # If there is no relation information,
    # use entity overlap only as a weak signal.
    #
    # IMPORTANT:
    # Entity overlap alone must NEVER produce SUPPORT.
    # --------------------------------------------------------

    if (
        claim_relation is None
        or evidence_relation is None
    ):

        if entity_score == 0:
            verdict = "NEUTRAL"

        else:
            verdict = "NEUTRAL"

    # --------------------------------------------------------
    # Relation score
    # --------------------------------------------------------

    if verdict == "SUPPORT":
        relation_score = 1.0

    elif verdict == "CONTRADICT":
        relation_score = 1.0

    elif verdict == "NOT_SUPPORT":
        relation_score = 0.0

    else:
        relation_score = 0.0

    # --------------------------------------------------------
    # Overall consistency score
    #
    # This is a diagnostic score.
    # The categorical verdict is more important.
    # --------------------------------------------------------

    consistency_score = (
        0.40 * entity_score
        + 0.60 * relation_score
    )

    return {
        "claim": claim,
        "evidence": evidence,

        "claim_entities": claim_entities,
        "evidence_entities": evidence_entities,

        "claim_relation": claim_relation,
        "evidence_relation": evidence_relation,

        "entity_consistency": round(
            entity_score,
            4
        ),

        "relation_consistency": round(
            relation_score,
            4
        ),

        "consistency_score": round(
            consistency_score,
            4
        ),

        "verdict": verdict,
    }


# ============================================================
# FIXED TEST SUITE
# ============================================================

def run_tests():

    tests = [

        {
            "claim":
                "India's capital city is New Delhi.",

            "evidence":
                "New Delhi is national capital of India.",

            "expected":
                "SUPPORT",
        },

        {
            "claim":
                "India's capital city is Mumbai.",

            "evidence":
                "New Delhi is national capital of India.",

            "expected":
                "CONTRADICT",
        },

        {
            "claim":
                "Mumbai is India's capital.",

            "evidence":
                "Mumbai is the capital of Maharashtra.",

            "expected":
                "NOT_SUPPORT",
        },

        {
            "claim":
                "Mumbai is Maharashtra's capital.",

            "evidence":
                "Mumbai is the capital of Maharashtra.",

            "expected":
                "SUPPORT",
        },

        {
            "claim":
                "India's capital is Mumbai.",

            "evidence":
                "Mumbai is the capital of Maharashtra.",

            "expected":
                "NOT_SUPPORT",
        },

        {
            "claim":
                "India's capital is Mumbai.",

            "evidence":
                "Mumbai is commercial capital of India.",

            "expected":
                "NOT_SUPPORT",
        },

        {
            "claim":
                "India's capital is Mumbai.",

            "evidence":
                "India is the largest country in Asia.",

            "expected":
                "NEUTRAL",
        },

        {
            "claim":
                "India banned X.",

            "evidence":
                "India did not ban X.",

            "expected":
                "CONTRADICT",
        },
    ]

    print("=" * 70)
    print(
        "ENTITY / RELATION CONSISTENCY TEST SUITE"
    )
    print("=" * 70)

    passed = 0

    for i, test in enumerate(
        tests,
        start=1
    ):

        result = check_consistency(
            test["claim"],
            test["evidence"]
        )

        actual = result["verdict"]
        expected = test["expected"]

        if actual == expected:

            status = "PASS"
            passed += 1

        else:

            status = "FAIL"

        print(f"\nTEST {i}: {status}")
        print("-" * 70)

        print("Claim:")
        print(test["claim"])

        print("\nEvidence:")
        print(test["evidence"])

        print("\nClaim entities:")
        print(
            result["claim_entities"]
        )

        print("Evidence entities:")
        print(
            result["evidence_entities"]
        )

        print("\nClaim relation:")
        print(
            result["claim_relation"]
        )

        print("Evidence relation:")
        print(
            result["evidence_relation"]
        )

        print(
            "\nEntity consistency:",
            result["entity_consistency"]
        )

        print(
            "Relation consistency:",
            result["relation_consistency"]
        )

        print(
            "Consistency score:",
            result["consistency_score"]
        )

        print(
            "Expected:",
            expected
        )

        print(
            "Actual:",
            actual
        )

    print("\n" + "=" * 70)

    print(
        f"RESULT: {passed}/{len(tests)} tests passed"
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_tests()