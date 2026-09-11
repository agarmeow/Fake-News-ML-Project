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
print("DUPLICATE AND DATA QUALITY ANALYSIS")
print("=" * 60)


# ==============================
# 1. DUPLICATE TEXTS
# ==============================

duplicate_mask = df["text"].duplicated(keep=False)

duplicates = df[duplicate_mask].copy()

print("\nTotal rows involved in duplicate texts:")
print(len(duplicates))

print("\nNumber of unique duplicated texts:")
print(duplicates["text"].nunique())


# ==============================
# 2. DUPLICATES BY LABEL
# ==============================

print("\nDuplicate texts grouped by label:")

duplicate_label_counts = (
    duplicates
    .groupby("text")["label"]
    .nunique()
)

same_label_duplicates = (duplicate_label_counts == 1).sum()
different_label_duplicates = (duplicate_label_counts > 1).sum()

print("Duplicate texts with SAME label:", same_label_duplicates)
print("Duplicate texts with DIFFERENT labels:", different_label_duplicates)


# ==============================
# 3. DUPLICATES BY LANGUAGE
# ==============================

print("\nDuplicate texts with different languages:")

duplicate_language_counts = (
    duplicates
    .groupby("text")["language"]
    .nunique()
)

cross_language_duplicates = (
    duplicate_language_counts > 1
).sum()

print(
    "Unique duplicated texts appearing in multiple languages:",
    cross_language_duplicates
)


# ==============================
# 4. SHOW EXAMPLES
# ==============================

print("\n" + "=" * 60)
print("DUPLICATE EXAMPLES")
print("=" * 60)

shown = 0

for text, group in duplicates.groupby("text"):

    if len(group) < 2:
        continue

    print("\n--- DUPLICATE ---")

    print("Occurrences:", len(group))

    print(
        group[
            ["id", "language", "label", "source_file"]
        ].to_string(index=False)
    )

    print("\nText preview:")
    print(text[:500])

    shown += 1

    if shown >= 5:
        break


# ==============================
# 5. EMPTY BODY
# ==============================

print("\n" + "=" * 60)
print("EMPTY BODY ARTICLES")
print("=" * 60)

empty_body = df[
    df["body"].isna() |
    (df["body"].fillna("").str.strip() == "")
]

print("Number of empty-body articles:", len(empty_body))

if len(empty_body) > 0:
    print(
        empty_body[
            ["id", "language", "label", "title", "source_file"]
        ].to_string(index=False)
    )


# ==============================
# 6. VERY SHORT ARTICLES
# ==============================

print("\n" + "=" * 60)
print("VERY SHORT ARTICLES")
print("=" * 60)

df["text_length"] = df["text"].fillna("").str.len()

short = df[df["text_length"] < 50]

print("Articles shorter than 50 characters:", len(short))

if len(short) > 0:
    print(
        short[
            ["id", "language", "label", "title", "text_length"]
        ].head(20).to_string(index=False)
    )


print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)