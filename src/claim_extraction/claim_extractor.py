import spacy
import re


# Load English NLP pipeline
nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    """Basic text cleaning."""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_claims(text):
    """
    Extract candidate factual claims from an article.

    This is a first-stage rule-based candidate extractor.
    It identifies sentences that contain potentially
    verifiable statements.
    """

    text = clean_text(text)

    doc = nlp(text)

    claims = []

    for sentence in doc.sents:

        sentence_text = sentence.text.strip()

        if not sentence_text:
            continue

        # Ignore extremely short sentences
        if len(sentence_text.split()) < 5:
            continue

        # Ignore questions
        if sentence_text.endswith("?"):
            continue

        # Ignore obvious headings
        if len(sentence_text.split()) <= 8 and sentence_text.isupper():
            continue

        # A sentence containing a verb is a reasonable
        # candidate for a factual claim.
        has_verb = any(
            token.pos_ in {"VERB", "AUX"}
            for token in sentence
        )

        if has_verb:
            claims.append(sentence_text)

    return claims


if __name__ == "__main__":

    print("=" * 60)
    print("CLAIM EXTRACTION TEST")
    print("=" * 60)

    article = """
    The Indian government announced a new policy for digital education on Monday.
    The policy will provide free online courses to students across the country.
    According to the announcement, more than 10 million students will benefit from the program.
    The initiative is scheduled to begin in October 2026.
    Some experts believe the policy could significantly improve access to education.
    The program has already received support from several education organizations.
    """

    print("\nINPUT ARTICLE:")
    print(article.strip())

    claims = extract_claims(article)

    print("\nEXTRACTED CLAIMS:")
    print("-" * 60)

    for i, claim in enumerate(claims, start=1):
        print(f"{i}. {claim}")

    print(f"\nTotal candidate claims: {len(claims)}")