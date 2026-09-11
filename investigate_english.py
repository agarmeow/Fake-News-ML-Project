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

FAKE_FILE = os.path.join(
    ENGLISH_DIR,
    "Fake.csv"
)

REAL_FILE = os.path.join(
    ENGLISH_DIR,
    "True.csv"
)


# ============================================================
# LOAD
# ============================================================

fake = pd.read_csv(FAKE_FILE)
real = pd.read_csv(REAL_FILE)

print("=" * 60)
print("ENGLISH DATASET QUALITY INVESTIGATION")
print("=" * 60)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    return " ".join(
        str(text).lower().split()
    )


fake["normalized_text"] = fake["text"].map(normalize)
real["normalized_text"] = real["text"].map(normalize)


# ============================================================
# TEXT DUPLICATION
# ============================================================

print("\n" + "=" * 60)
print("TEXT DUPLICATION")
print("=" * 60)

fake_text_counts = fake["normalized_text"].value_counts()
real_text_counts = real["normalized_text"].value_counts()

print(
    "\nFake unique texts:",
    fake["normalized_text"].nunique()
)

print(
    "Real unique texts:",
    real["normalized_text"].nunique()
)

print(
    "\nFake texts appearing more than once:",
    (fake_text_counts > 1).sum()
)

print(
    "Real texts appearing more than once:",
    (real_text_counts > 1).sum()
)


# ============================================================
# CROSS-LABEL DUPLICATES
# ============================================================

fake_texts = set(fake["normalized_text"])
real_texts = set(real["normalized_text"])

common_texts = fake_texts.intersection(real_texts)

# Remove empty text
common_texts.discard("")

print("\n" + "=" * 60)
print("CROSS-LABEL DUPLICATES")
print("=" * 60)

print(
    "\nTexts appearing in BOTH Fake and Real:",
    len(common_texts)
)


# ============================================================
# SHOW CROSS-LABEL EXAMPLES
# ============================================================

if len(common_texts) > 0:

    print("\nExamples:")

    shown = 0

    for text in common_texts:

        fake_rows = fake[
            fake["normalized_text"] == text
        ]

        real_rows = real[
            real["normalized_text"] == text
        ]

        print("\n----------------------------------------")

        print("\nFAKE:")
        print(
            fake_rows[
                ["title", "subject", "date"]
            ].head(3).to_string(index=False)
        )

        print("\nREAL:")
        print(
            real_rows[
                ["title", "subject", "date"]
            ].head(3).to_string(index=False)
        )

        print("\nText preview:")
        print(text[:500])

        shown += 1

        if shown >= 5:
            break


# ============================================================
# VERY SHORT ARTICLES
# ============================================================

fake["text_length"] = fake["text"].fillna("").astype(str).str.len()
real["text_length"] = real["text"].fillna("").astype(str).str.len()

fake_short = fake[
    fake["text_length"] < 50
]

real_short = real[
    real["text_length"] < 50
]

print("\n" + "=" * 60)
print("VERY SHORT ARTICLES")
print("=" * 60)

print(
    "\nFake articles shorter than 50 characters:",
    len(fake_short)
)

print(
    "Real articles shorter than 50 characters:",
    len(real_short)
)


# ============================================================
# SHOW SHORT FAKE ARTICLES
# ============================================================

if len(fake_short) > 0:

    print("\nShort Fake examples:")

    print(
        fake_short[
            ["title", "text", "subject", "date", "text_length"]
        ]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# SHOW SHORT REAL ARTICLES
# ============================================================

if len(real_short) > 0:

    print("\nShort Real examples:")

    print(
        real_short[
            ["title", "text", "subject", "date", "text_length"]
        ]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("SUBJECT DISTRIBUTION")
print("=" * 60)

print("\nFake subjects:")
print(fake["subject"].value_counts())

print("\nReal subjects:")
print(real["subject"].value_counts())


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("INVESTIGATION COMPLETE")
print("=" * 60)