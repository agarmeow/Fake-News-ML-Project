import os
import random
import re
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXTRACTED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "extracted"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

SAMPLES_PER_CLASS = 707
RANDOM_SEED = 42

LANGUAGES = {
    "gujarati": "GU",
    "hindi": "HI",
    "marathi": "MR",
    "telugu": "TE"
}

os.makedirs(PROCESSED_DIR, exist_ok=True)

random.seed(RANDOM_SEED)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Used only for detecting duplicates.
    We keep the original text unchanged in the final dataset.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Replace multiple spaces/newlines with one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    # Case normalization is useful for Latin text.
    # It does not affect Indic scripts.
    text = text.lower()

    return text


# ============================================================
# READ ARTICLE
# ============================================================

def read_article(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:
        content = f.read()

    # Remove empty lines
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    if not lines:
        return "", ""

    # First non-empty line = title
    title = lines[0]

    # Remaining lines = body
    body = " ".join(lines[1:]).strip()

    # Remove repeated title at beginning of body
    if body.startswith(title):
        body = body[len(title):].strip()

    return title, body


# ============================================================
# UNUSABLE TITLE CHECK
# ============================================================

def is_unusable_article(title, body):

    title_clean = normalize_text(title)
    body_clean = normalize_text(body)

    # Empty title
    if not title_clean:
        return True

    # Empty body
    if not body_clean:
        return True

    # Very short total text
    if len(title_clean + " " + body_clean) < 50:
        return True

    # Known "title unavailable" messages
    unavailable_titles = {

        # Hindi
        "शीर्षक उपलब्ध नहीं",
        "शीर्षक उपलब्ध नहीं है",

        # Marathi
        "शीर्षक उपलब्ध नाही",

        # Gujarati
        "હક ઉપલબ્ધ નથી",

        # Telugu
        "శీర్షిక అందుబాటులో లేదు"
    }

    if title_clean in unavailable_titles:
        return True

    return False


# ============================================================
# PROCESS ONE LANGUAGE
# ============================================================

def process_language(language, language_code):

    language_dir = os.path.join(
        EXTRACTED_DIR,
        language
    )

    print("\n" + "=" * 60)
    print(f"PROCESSING: {language.upper()}")
    print("=" * 60)

    fake_files = []
    real_files = []

    # --------------------------------------------------------
    # FIND FILES
    # --------------------------------------------------------

    for root, dirs, files in os.walk(language_dir):

        # Ignore Mac metadata
        dirs[:] = [
            d for d in dirs
            if d != "__MACOSX"
        ]

        for filename in files:

            if not filename.lower().endswith(".txt"):
                continue

            file_path = os.path.join(
                root,
                filename
            )

            filename_lower = filename.lower()

            if "_fake_" in filename_lower:
                fake_files.append(file_path)

            elif "_real_" in filename_lower:
                real_files.append(file_path)

    print(f"Total fake files: {len(fake_files)}")
    print(f"Total real files: {len(real_files)}")

    records = []

    # ========================================================
    # PROCESS FAKE AND REAL
    # ========================================================

    for label, files in [
        (0, fake_files),
        (1, real_files)
    ]:

        usable_articles = []
        seen_texts = set()

        skipped_empty = 0
        skipped_short = 0
        skipped_duplicate = 0

        for file_path in files:

            title, body = read_article(file_path)

            # ------------------------------------------------
            # CHECK EMPTY / UNUSABLE
            # ------------------------------------------------

            if not title or not body:
                skipped_empty += 1
                continue

            # ------------------------------------------------
            # CHECK SHORT / UNAVAILABLE
            # ------------------------------------------------

            if is_unusable_article(title, body):
                skipped_short += 1
                continue

            # ------------------------------------------------
            # COMBINE TITLE + BODY
            # ------------------------------------------------

            text = f"{title} {body}".strip()

            # ------------------------------------------------
            # DUPLICATE DETECTION
            # ------------------------------------------------

            normalized = normalize_text(text)

            if normalized in seen_texts:
                skipped_duplicate += 1
                continue

            seen_texts.add(normalized)

            usable_articles.append(
                {
                    "title": title,
                    "body": body,
                    "text": text,
                    "label": label,
                    "source_file": os.path.basename(file_path)
                }
            )

        # ----------------------------------------------------
        # SHUFFLE
        # ----------------------------------------------------

        random.shuffle(usable_articles)

        # ----------------------------------------------------
        # TAKE EXACTLY 1000
        # ----------------------------------------------------

        selected = usable_articles[
            :SAMPLES_PER_CLASS
        ]

        print(
            f"\n{'FAKE' if label == 0 else 'REAL'}:"
        )

        print(
            f"  Usable unique articles: "
            f"{len(usable_articles)}"
        )

        print(
            f"  Selected: "
            f"{len(selected)}"
        )

        print(
            f"  Skipped empty: "
            f"{skipped_empty}"
        )

        print(
            f"  Skipped short/unusable: "
            f"{skipped_short}"
        )

        print(
            f"  Skipped duplicates: "
            f"{skipped_duplicate}"
        )

        # ----------------------------------------------------
        # ADD IDs
        # ----------------------------------------------------

        label_name = (
            "FAKE"
            if label == 0
            else "REAL"
        )

        for index, article in enumerate(
            selected,
            start=1
        ):

            article["id"] = (
                f"{language_code}_"
                f"{label_name}_"
                f"{index:05d}"
            )

            article["language"] = language

            records.append(article)

    return records


# ============================================================
# PROCESS ALL LANGUAGES
# ============================================================

all_records = []

for language, language_code in LANGUAGES.items():

    language_records = process_language(
        language,
        language_code
    )

    all_records.extend(language_records)


# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

df = pd.DataFrame(all_records)


# ============================================================
# SHUFFLE FINAL DATASET
# ============================================================

df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# REORDER COLUMNS
# ============================================================

df = df[
    [
        "id",
        "language",
        "title",
        "body",
        "text",
        "label",
        "source_file"
    ]
]


# ============================================================
# SAVE FINAL DATASET
# ============================================================

output_file = os.path.join(
    PROCESSED_DIR,
    "final_multilingual_fake_news.csv"
)

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("FINAL DATASET CREATED")
print("=" * 60)

print(
    f"\nTotal articles: {len(df)}"
)

print("\nLanguage distribution:")

print(
    df["language"].value_counts()
)

print("\nFake / Real distribution:")

print(
    df["label"].value_counts()
)

print("\nLanguage + Label distribution:")

print(
    df.groupby(
        ["language", "label"]
    )
    .size()
    .unstack(fill_value=0)
)

print("\nMissing values:")

print(
    df.isnull().sum()
)

print("\nDuplicate texts:")

print(
    df["text"].duplicated().sum()
)

print("\nSaved to:")

print(output_file)

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)