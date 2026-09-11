import os
import re
import random
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENGLISH_DIR = os.path.join(
    BASE_DIR,
    "data",
    "external",
    "english"
)

INDIAN_DATASET = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_multilingual_fake_news.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_5_language_dataset.csv"
)

SAMPLES_PER_CLASS = 707
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


# ============================================================
# LOAD ENGLISH DATA
# ============================================================

print("=" * 60)
print("ADDING ENGLISH DATASET")
print("=" * 60)

fake_file = os.path.join(
    ENGLISH_DIR,
    "Fake.csv"
)

real_file = os.path.join(
    ENGLISH_DIR,
    "True.csv"
)

fake = pd.read_csv(fake_file)
real = pd.read_csv(real_file)

print("\nOriginal English data:")
print("Fake:", len(fake))
print("Real:", len(real))


# ============================================================
# CLEAN ENGLISH DATA
# ============================================================

def clean_english(df, label):

    df = df.copy()

    # --------------------------------------------------------
    # Remove missing title/text
    # --------------------------------------------------------

    df = df.dropna(
        subset=["title", "text"]
    )

    # --------------------------------------------------------
    # Convert to strings
    # --------------------------------------------------------

    df["title"] = df["title"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    # --------------------------------------------------------
    # Remove extremely short article bodies
    # --------------------------------------------------------

    df["text_length"] = df["text"].str.len()

    before_short = len(df)

    df = df[
        df["text_length"] >= 50
    ].copy()

    removed_short = (
        before_short - len(df)
    )

    # --------------------------------------------------------
    # Normalize article body for duplicate detection
    # --------------------------------------------------------

    df["normalized_body"] = (
        df["text"].map(normalize_text)
    )

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["normalized_body"],
        keep="first"
    ).copy()

    removed_duplicates = (
        before_duplicates - len(df)
    )

    # --------------------------------------------------------
    # Remove empty normalized text
    # --------------------------------------------------------

    df = df[
        df["normalized_body"].str.len() > 0
    ].copy()

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Select required number
    # --------------------------------------------------------

    selected = df.head(
        SAMPLES_PER_CLASS
    ).copy()

    print(
        f"\n{'FAKE' if label == 0 else 'REAL'}"
    )

    print(
        "After removing short articles:",
        before_duplicates
    )

    print(
        "Removed short articles:",
        removed_short
    )

    print(
        "Removed duplicate bodies:",
        removed_duplicates
    )

    print(
        "Unique usable articles:",
        len(df)
    )

    print(
        "Selected:",
        len(selected)
    )

    return selected


# ============================================================
# CLEAN + SELECT
# ============================================================

fake = clean_english(
    fake,
    label=0
)

real = clean_english(
    real,
    label=1
)


# ============================================================
# CONVERT ENGLISH FORMAT
# ============================================================

english_records = []


for index, row in fake.iterrows():

    title = row["title"]
    body = row["text"]

    english_records.append({

        "id": f"EN_FAKE_{index + 1:05d}",

        "language": "english",

        "title": title,

        "body": body,

        "text": f"{title} {body}".strip(),

        "label": 0,

        "source_file": "Fake.csv"

    })


for index, row in real.iterrows():

    title = row["title"]
    body = row["text"]

    english_records.append({

        "id": f"EN_REAL_{index + 1:05d}",

        "language": "english",

        "title": title,

        "body": body,

        "text": f"{title} {body}".strip(),

        "label": 1,

        "source_file": "True.csv"

    })


english_df = pd.DataFrame(
    english_records
)


# ============================================================
# LOAD INDIAN DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING INDIAN DATASET")
print("=" * 60)

indian_df = pd.read_csv(
    INDIAN_DATASET
)

print(
    "\nIndian articles:",
    len(indian_df)
)


# ============================================================
# COMBINE
# ============================================================

final_df = pd.concat(
    [
        indian_df,
        english_df
    ],
    ignore_index=True
)


# ============================================================
# FINAL DUPLICATE CHECK
# ============================================================

final_df["normalized_text"] = (
    final_df["text"]
    .map(normalize_text)
)

duplicate_count = (
    final_df["normalized_text"]
    .duplicated()
    .sum()
)

print("\n" + "=" * 60)
print("FINAL DATASET CHECK")
print("=" * 60)

print(
    "\nDuplicate texts after combining:",
    duplicate_count
)


# ============================================================
# REMOVE NORMALIZATION HELPER COLUMN
# ============================================================

final_df = final_df.drop(
    columns=["normalized_text"]
)


# ============================================================
# SHUFFLE
# ============================================================

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\nTotal articles:")
print(len(final_df))

print("\nLanguage distribution:")
print(
    final_df["language"].value_counts()
)

print("\nFake / Real distribution:")
print(
    final_df["label"].value_counts()
)

print("\nLanguage + Label:")
print(
    final_df
    .groupby(["language", "label"])
    .size()
    .unstack(fill_value=0)
)

print("\nMissing values:")
print(
    final_df.isnull().sum()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("5-LANGUAGE DATASET COMPLETE")
print("=" * 60)