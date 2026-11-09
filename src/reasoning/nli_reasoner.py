import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def load_nli_model():
    print("\nLoading NLI model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    # Check for NaN on a dummy sample; fallback to float32 if NaN
    if device == "cuda":
        dummy = tokenizer("test premise", "test hypothesis", return_tensors="pt", max_length=16, truncation=True).to(device)
        with torch.inference_mode():
            out = model(**dummy)
        if torch.isnan(out.logits).any():
            print("Warning: NaN detected in fp16, falling back to float32")
            model = model.to(dtype=torch.float32)

    return tokenizer, model


def classify_claim_evidence_batch(
    claim,
    evidence_list,
    tokenizer,
    model
):
    """
    Perform Natural Language Inference on a batch of evidence passages for a claim.

    Premise = evidence passage
    Hypothesis = claim
    """
    if not evidence_list:
        return []

    device = next(model.parameters()).device
    inputs = tokenizer(
        evidence_list,
        [claim] * len(evidence_list),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits

        # If NaN detected, fallback to float32
        if torch.isnan(logits).any():
            model_fp32 = model.to(dtype=torch.float32)
            outputs = model_fp32(**inputs)
            logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)

    results = []
    for prob in probabilities:
        label_scores = {}
        for idx, score in enumerate(prob):
            label = model.config.id2label[idx].lower()
            label_scores[label] = float(score)

        predicted_id = torch.argmax(prob).item()
        predicted_label = model.config.id2label[predicted_id]
        confidence = prob[predicted_id].item()

        results.append({
            "label": predicted_label,
            "confidence": confidence,
            "scores": label_scores
        })

    return results


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
    return classify_claim_evidence_batch(claim, [evidence], tokenizer, model)[0]


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
