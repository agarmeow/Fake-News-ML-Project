import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

DATA_DIR = "models/clip"

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_split(split):

    text = np.load(
        f"{DATA_DIR}/{split}_text.npy"
    )

    image = np.load(
        f"{DATA_DIR}/{split}_image.npy"
    )

    labels = np.load(
        f"{DATA_DIR}/{split}_labels.npy"
    )

    return (
        torch.tensor(text, dtype=torch.float32),
        torch.tensor(image, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )


train_text, train_image, train_y = load_split("train")
val_text, val_image, val_y = load_split("validate")
test_text, test_image, test_y = load_split("test")


print("\nLoaded:")
print("Train:", len(train_y))
print("Validation:", len(val_y))
print("Test:", len(test_y))

TEXT_DIM = train_text.shape[1]
IMAGE_DIM = train_image.shape[1]

print("Text embedding dimension:", TEXT_DIM)
print("Image embedding dimension:", IMAGE_DIM)


# ============================================================
# CLASSIFIER
# ============================================================

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


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    train_x,
    train_y,
    val_x,
    val_y,
    model_name
):

    model = Classifier(
        train_x.shape[1]
    ).to(device)

    dataset = TensorDataset(
        train_x,
        train_y
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0

    print(f"\nTraining {model_name}...")

    for epoch in range(EPOCHS):

        model.train()

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output,
                y
            )

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():

            val_output = model(
                val_x.to(device)
            )

            val_pred = torch.argmax(
                val_output,
                dim=1
            ).cpu()

        val_acc = accuracy_score(
            val_y.numpy(),
            val_pred.numpy()
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Validation Accuracy: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                f"{DATA_DIR}/{model_name}.pt"
            )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    test_x,
    test_y,
    name
):

    model.eval()

    with torch.no_grad():

        output = model(
            test_x.to(device)
        )

        predictions = torch.argmax(
            output,
            dim=1
        ).cpu().numpy()

    y_true = test_y.numpy()

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

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro-F1 : {f1:.4f}")


# ============================================================
# 1. TEXT-ONLY
# ============================================================

text_model = train_model(
    train_text,
    train_y,
    val_text,
    val_y,
    "text_only"
)

evaluate(
    text_model,
    test_text,
    test_y,
    "TEXT-ONLY"
)


# ============================================================
# 2. IMAGE-ONLY
# ============================================================

image_model = train_model(
    train_image,
    train_y,
    val_image,
    val_y,
    "image_only"
)

evaluate(
    image_model,
    test_image,
    test_y,
    "IMAGE-ONLY"
)


# ============================================================
# 3. MULTIMODAL
# ============================================================

train_multimodal = torch.cat(
    [train_text, train_image],
    dim=1
)

val_multimodal = torch.cat(
    [val_text, val_image],
    dim=1
)

test_multimodal = torch.cat(
    [test_text, test_image],
    dim=1
)

multimodal_model = train_model(
    train_multimodal,
    train_y,
    val_multimodal,
    val_y,
    "multimodal"
)

evaluate(
    multimodal_model,
    test_multimodal,
    test_y,
    "TEXT + IMAGE MULTIMODAL"
)


print("\nAll three experiments completed.")