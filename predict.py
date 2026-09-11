import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/muril_classifier"
MAX_LENGTH = 256

print("Loading MuRIL model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Using device: {device}")

print("\nMuRIL Fake News Classifier")
print("=" * 50)

while True:

    text = input("\nEnter an article/headline (or type 'exit'): ")

    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    prediction = torch.argmax(probabilities).item()

    # Dataset labels:
    # 0 = Real
    # 1 = Fake

    label = "FAKE" if prediction == 1 else "REAL"
    confidence = probabilities[prediction].item() * 100

    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}%")