from transformers import pipeline


# Multilingual zero-shot classifier
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


print("Loading multilingual claim classifier...")

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME
)

print("Model loaded successfully.")


def classify_claims(sentences):
    """
    Classify sentences as factual claims or non-factual statements.
    """

    candidate_labels = [
        "factual claim",
        "opinion or non-factual statement"
    ]

    results = []

    for sentence in sentences:

        result = classifier(
            sentence,
            candidate_labels
        )

        predicted_label = result["labels"][0]
        confidence = result["scores"][0]

        results.append({
            "sentence": sentence,
            "label": predicted_label,
            "confidence": confidence
        })

    return results


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("MULTILINGUAL CLAIM EXTRACTION TEST")
    print("=" * 60)

    test_sentences = [

        # English
        "India's capital city is New Delhi.",

        # Hindi
        "भारत की राजधानी नई दिल्ली है।",

        # Gujarati
        "ભારતની રાજધાની નવી દિલ્હી છે.",

        # Marathi
        "भारताची राजधानी नवी दिल्ली आहे.",

        # Telugu
        "భారతదేశ రాజధాని న్యూఢిల్లీ."
    ]

    results = classify_claims(test_sentences)

    print("\nRESULTS:")
    print("-" * 60)

    for result in results:

        print(f"\nSentence: {result['sentence']}")
        print(f"Classification: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")