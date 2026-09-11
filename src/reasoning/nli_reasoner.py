import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def load_nli_model():
    print("\nLoading NLI model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


def classify_claim_evidence(
    claim,
    evidence,
    tokenizer,
    model
):
    """
    Perform Natural Language Inference.

    Premise = evidence
    Hypothesis = claim
    """

    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    label_scores = {}

    for idx, score in enumerate(probabilities):
        label = model.config.id2label[idx].lower()
        label_scores[label] = float(score)

    predicted_id = torch.argmax(probabilities).item()

    predicted_label = model.config.id2label[
        predicted_id
    ]

    confidence = probabilities[
        predicted_id
    ].item()

    return {
        "label": predicted_label,
        "confidence": confidence,
        "scores": label_scores
    }


def main():

    claim = input(
        "\nEnter claim: "
    ).strip()

    if not claim:
        print("No claim entered.")
        return

    tokenizer, model = load_nli_model()

    evidence = input(
        "\nEnter evidence passage: "
    ).strip()

    if not evidence:
        print("No evidence entered.")
        return

    result = classify_claim_evidence(
        claim,
        evidence,
        tokenizer,
        model
    )

    print("\n" + "=" * 70)
    print("CLAIM-EVIDENCE REASONING")
    print("=" * 70)

    print(f"\nClaim:\n{claim}")

    print(f"\nEvidence:\n{evidence}")

    print(
        f"\nNLI Result: {result['label']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.4f}"
    )

    print("\nAll NLI probabilities:")

    for label, score in result["scores"].items():
        print(
            f"{label}: {score:.4f}"
        )


if __name__ == "__main__":
    main()