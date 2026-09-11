import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# Paths
# =========================

DATA_DIR = Path("data/processed/splits")

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

# =========================
# Load datasets
# =========================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")

# =========================
# Prepare text
# =========================

# Model input = title + article body
train_text = (
    train_df["title"].fillna("") + " " +
    train_df["body"].fillna("")
)

val_text = (
    val_df["title"].fillna("") + " " +
    val_df["body"].fillna("")
)

test_text = (
    test_df["title"].fillna("") + " " +
    test_df["body"].fillna("")
)

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

# =========================
# TF-IDF
# =========================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_text)
X_val = vectorizer.transform(val_text)
X_test = vectorizer.transform(test_text)

print(f"Training features:   {X_train.shape}")
print(f"Validation features: {X_val.shape}")
print(f"Testing features:    {X_test.shape}")

# =========================
# Logistic Regression
# =========================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Validation
# =========================

print("\nEvaluating on validation set...")

val_pred = model.predict(X_val)

print(f"Validation Accuracy:  {accuracy_score(y_val, val_pred):.4f}")
print(f"Validation Precision: {precision_score(y_val, val_pred):.4f}")
print(f"Validation Recall:    {recall_score(y_val, val_pred):.4f}")
print(f"Validation F1:        {f1_score(y_val, val_pred):.4f}")

# =========================
# Final Test Evaluation
# =========================

print("\nEvaluating on test set...")

test_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)

print("\n" + "=" * 50)
print("FINAL TEST RESULTS")
print("=" * 50)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Real", "Fake"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# =========================
# Per-language evaluation
# =========================

print("\n" + "=" * 50)
print("PER-LANGUAGE TEST RESULTS")
print("=" * 50)

test_results = test_df.copy()
test_results["prediction"] = test_pred

for language in sorted(test_results["language"].unique()):
    lang_df = test_results[test_results["language"] == language]

    lang_accuracy = accuracy_score(
        lang_df["label"],
        lang_df["prediction"]
    )

    lang_precision = precision_score(
        lang_df["label"],
        lang_df["prediction"],
        zero_division=0
    )

    lang_recall = recall_score(
        lang_df["label"],
        lang_df["prediction"],
        zero_division=0
    )

    lang_f1 = f1_score(
        lang_df["label"],
        lang_df["prediction"],
        zero_division=0
    )

    print(f"\n{language.upper()}")
    print(f"Samples:   {len(lang_df)}")
    print(f"Accuracy:  {lang_accuracy:.4f}")
    print(f"Precision: {lang_precision:.4f}")
    print(f"Recall:    {lang_recall:.4f}")
    print(f"F1 Score:  {lang_f1:.4f}")