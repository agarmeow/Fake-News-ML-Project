import os
import pandas as pd

# ==============================
# LOAD DATASET
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "multilingual_fake_news.csv"
)

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("DATASET QUALITY CHECK")
print("=" * 60)

# ==============================
# BASIC INFORMATION
# ==============================

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ==============================
# MISSING VALUES
# ==============================

print("\nMissing values:")
print(df.isnull().sum())

# ==============================
# DUPLICATES
# ==============================

print("\nDuplicate complete rows:")
print(df.duplicated().sum())

print("\nDuplicate texts:")
print(df["text"].duplicated().sum())

# ==============================
# TEXT LENGTH
# ==============================

df["text_length"] = df["text"].astype(str).str.len()

print("\nText length statistics:")
print(df["text_length"].describe())

# ==============================
# VERY SHORT ARTICLES
# ==============================

short_articles = df[df["text_length"] < 50]

print("\nArticles shorter than 50 characters:")
print(len(short_articles))

# ==============================
# LANGUAGE DISTRIBUTION
# ==============================

print("\nLanguage distribution:")
print(df["language"].value_counts())

# ==============================
# LABEL DISTRIBUTION
# ==============================

print("\nLabel distribution:")
print(df["label"].value_counts())

# ==============================
# LANGUAGE + LABEL
# ==============================

print("\nLanguage + Label distribution:")
print(
    df.groupby(["language", "label"])
      .size()
      .unstack(fill_value=0)
)

# ==============================
# EMPTY TITLES / BODIES
# ==============================

print("\nEmpty titles:")
print(
    (df["title"].fillna("").str.strip() == "").sum()
)

print("\nEmpty bodies:")
print(
    (df["body"].fillna("").str.strip() == "").sum()
)

# ==============================
# SAMPLE ARTICLES
# ==============================

print("\n" + "=" * 60)
print("SAMPLE ARTICLES")
print("=" * 60)

for language in ["gujarati", "hindi", "marathi", "telugu"]:

    sample = df[df["language"] == language].iloc[0]

    print(f"\n--- {language.upper()} ---")

    print("ID:", sample["id"])
    print("Label:", "FAKE" if sample["label"] == 0 else "REAL")
    print("Title:", sample["title"])

    print("Body:")
    print(str(sample["body"])[:500])

print("\n" + "=" * 60)
print("QUALITY CHECK COMPLETE")
print("=" * 60)