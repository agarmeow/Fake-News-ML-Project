import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

DATA_DIR = "models/clip"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class Classifier(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.network(x)


def load_data(split):

    text = torch.tensor(
        np.load(f"{DATA_DIR}/{split}_text.npy"),
        dtype=torch.float32
    )

    image = torch.tensor(
        np.load(f"{DATA_DIR}/{split}_image.npy"),
        dtype=torch.float32
    )

    labels = torch.tensor(
        np.load(f"{DATA_DIR}/{split}_labels.npy"),
        dtype=torch.long
    )

    return text, image, labels


def evaluate_model(model_name, x, y):

    model = Classifier(x.shape[1]).to(device)

    model.load_state_dict(
        torch.load(
            f"{DATA_DIR}/{model_name}.pt",
            map_location=device
        )
    )

    model.eval()

    with torch.no_grad():

        output = model(x.to(device))

        predictions = torch.argmax(
            output,
            dim=1
        ).cpu().numpy()

    y_true = y.numpy()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro-F1 : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    print("\nPer-class results:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Class 0", "Class 1"],
            digits=4,
            zero_division=0
        )
    )

    return predictions


# Load test data
test_text, test_image, test_y = load_data("test")

# Text-only
text_predictions = evaluate_model(
    "text_only",
    test_text,
    test_y
)

# Image-only
image_predictions = evaluate_model(
    "image_only",
    test_image,
    test_y
)

# Multimodal
test_multimodal = torch.cat(
    [test_text, test_image],
    dim=1
)

multimodal_predictions = evaluate_model(
    "multimodal",
    test_multimodal,
    test_y
)

print("\nEvaluation complete.")