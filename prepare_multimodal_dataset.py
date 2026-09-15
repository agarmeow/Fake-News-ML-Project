import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

INPUT = Path("data/multimodal_train.tsv")
OUTPUT_DIR = Path("data/multimodal_clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(INPUT, sep="\t")

# Keep ONLY what our experiment is allowed to use
df = df[["clean_title", "image_url", "2_way_label"]].copy()

# Remove missing/empty values
df = df.dropna(subset=["clean_title", "image_url", "2_way_label"])
df["clean_title"] = df["clean_title"].astype(str).str.strip()
df["image_url"] = df["image_url"].astype(str).str.strip()
df = df[(df["clean_title"] != "") & (df["image_url"] != "")]

# Remove exact duplicate text+image pairs
before = len(df)
df = df.drop_duplicates(subset=["clean_title", "image_url"])
print(f"Removed exact duplicates: {before - len(df)}")

# Make a balanced 20,000-sample dataset
class_0 = df[df["2_way_label"] == 0]
class_1 = df[df["2_way_label"] == 1]

N_PER_CLASS = 10_000

if len(class_0) < N_PER_CLASS or len(class_1) < N_PER_CLASS:
    raise ValueError("Not enough samples in one of the classes.")

class_0 = class_0.sample(N_PER_CLASS, random_state=42)
class_1 = class_1.sample(N_PER_CLASS, random_state=42)

df = pd.concat([class_0, class_1])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train / validation / test = 80 / 10 / 10
train, temp = train_test_split(
    df,
    test_size=0.20,
    stratify=df["2_way_label"],
    random_state=42
)

validate, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["2_way_label"],
    random_state=42
)

# Save
train.to_csv(OUTPUT_DIR / "train.tsv", sep="\t", index=False)
validate.to_csv(OUTPUT_DIR / "validate.tsv", sep="\t", index=False)
test.to_csv(OUTPUT_DIR / "test.tsv", sep="\t", index=False)

print("\nCreated:")
print(f"Train:      {len(train)}")
print(f"Validation: {len(validate)}")
print(f"Test:       {len(test)}")

print("\nLabel distribution:")
print("Train:\n", train["2_way_label"].value_counts())
print("Validation:\n", validate["2_way_label"].value_counts())
print("Test:\n", test["2_way_label"].value_counts())