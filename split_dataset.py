import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# =========================
# Configuration
# =========================

INPUT_FILE = Path("data/processed/final_5_language_dataset.csv")
OUTPUT_DIR = Path("data/processed/splits")

RANDOM_SEED = 42

# =========================
# Load dataset
# =========================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total records: {len(df)}")

# =========================
# First split: 70% train, 30% temporary
# =========================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df[["language", "label"]]
)

# =========================
# Second split:
# 15% validation, 15% test
# =========================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df[["language", "label"]]
)

# =========================
# Create output directory
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Save splits
# =========================

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
validation_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False,
    encoding="utf-8-sig"
)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

# =========================
# Verification
# =========================

print("\nSplit completed.")
print("-" * 50)

print(f"Train:      {len(train_df)}")
print(f"Validation: {len(validation_df)}")
print(f"Test:       {len(test_df)}")
print(f"Total:      {len(train_df) + len(validation_df) + len(test_df)}")

print("\nOverall label distribution:")
print("\nTRAIN")
print(train_df["label"].value_counts())

print("\nVALIDATION")
print(validation_df["label"].value_counts())

print("\nTEST")
print(test_df["label"].value_counts())

print("\nLanguage distribution:")
print("\nTRAIN")
print(train_df["language"].value_counts())

print("\nVALIDATION")
print(validation_df["language"].value_counts())

print("\nTEST")
print(test_df["language"].value_counts())

print("\nFiles created:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "validation.csv")
print(OUTPUT_DIR / "test.csv")