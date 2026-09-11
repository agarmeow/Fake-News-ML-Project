import os
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


# ============================================================
# FILE PATHS
# ============================================================

FAKE_FILE = os.path.join(
    ENGLISH_DIR,
    "Fake.csv"
)

REAL_FILE = os.path.join(
    ENGLISH_DIR,
    "True.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ENGLISH DATASET INSPECTION")
print("=" * 60)

print("\nLoading Fake.csv...")

fake_df = pd.read_csv(FAKE_FILE)

print("Loading True.csv...")

real_df = pd.read_csv(REAL_FILE)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC INFORMATION")
print("=" * 60)

print("\nFake articles:", len(fake_df))
print("Real articles:", len(real_df))

print("\nFake columns:")
print(fake_df.columns.tolist())

print("\nReal columns:")
print(real_df.columns.tolist())


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print("\nFake:")
print(fake_df.dtypes)

print("\nReal:")
print(real_df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nFake:")
print(fake_df.isnull().sum())

print("\nReal:")
print(real_df.isnull().sum())


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print("\nFake duplicate rows:")
print(fake_df.duplicated().sum())

print("\nReal duplicate rows:")
print(real_df.duplicated().sum())


# ============================================================
# UNIQUE TEXTS
# ============================================================

if "text" in fake_df.columns:

    print("\nUnique Fake texts:")
    print(fake_df["text"].nunique())

if "text" in real_df.columns:

    print("\nUnique Real texts:")
    print(real_df["text"].nunique())


# ============================================================
# SAMPLE FAKE ARTICLE
# ============================================================

print("\n" + "=" * 60)
print("FAKE ARTICLE SAMPLE")
print("=" * 60)

print(fake_df.iloc[0].to_string())


# ============================================================
# SAMPLE REAL ARTICLE
# ============================================================

print("\n" + "=" * 60)
print("REAL ARTICLE SAMPLE")
print("=" * 60)

print(real_df.iloc[0].to_string())


# ============================================================
# TEXT LENGTH
# ============================================================

if "text" in fake_df.columns:

    fake_lengths = (
        fake_df["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print("\n" + "=" * 60)
    print("FAKE TEXT LENGTH")
    print("=" * 60)

    print(fake_lengths.describe())


if "text" in real_df.columns:

    real_lengths = (
        real_df["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print("\n" + "=" * 60)
    print("REAL TEXT LENGTH")
    print("=" * 60)

    print(real_lengths.describe())


print("\n" + "=" * 60)
print("ENGLISH INSPECTION COMPLETE")
print("=" * 60)