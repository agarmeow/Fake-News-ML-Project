import os
import random
import pandas as pd

# ==============================
# SETTINGS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR = os.path.join(BASE_DIR, "data", "extracted")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

SAMPLES_PER_CLASS = 1000
RANDOM_SEED = 42

LANGUAGES = {
    "gujarati": "GU",
    "hindi": "HI",
    "marathi": "MR",
    "telugu": "TE"
}

# Make sure processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

random.seed(RANDOM_SEED)


# ==============================
# READ ONE ARTICLE
# ==============================

def read_article(file_path):
    """
    Reads a .txt news file and separates:
    title + body
    """

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Remove empty lines and unnecessary whitespace
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    if not lines:
        return "", ""

    # First non-empty line is treated as title
    title = lines[0]

    # Everything after the title becomes the body
    body_lines = lines[1:]

    body = " ".join(body_lines).strip()

    # Some articles repeat the title at the beginning of the body.
    # Remove that exact duplicate once.
    if body.startswith(title):
        body = body[len(title):].strip()

    return title, body


# ==============================
# COLLECT FILES
# ==============================

all_records = []

for language, language_code in LANGUAGES.items():

    language_dir = os.path.join(EXTRACTED_DIR, language)

    if not os.path.exists(language_dir):
        print(f"ERROR: Folder not found: {language_dir}")
        continue

    print("\n" + "=" * 60)
    print(f"Processing language: {language.upper()}")
    print("=" * 60)

    fake_files = []
    real_files = []

    # Search all text files
    for root, dirs, files in os.walk(language_dir):

        # Ignore Mac metadata folders
        dirs[:] = [
            d for d in dirs
            if d != "__MACOSX"
        ]

        for filename in files:

            if not filename.lower().endswith(".txt"):
                continue

            file_path = os.path.join(root, filename)

            filename_lower = filename.lower()

            if "_fake_" in filename_lower:
                fake_files.append(file_path)

            elif "_real_" in filename_lower:
                real_files.append(file_path)

    print(f"Fake articles available: {len(fake_files)}")
    print(f"Real articles available: {len(real_files)}")

    # Check that enough files exist
    if len(fake_files) < SAMPLES_PER_CLASS:
        print(f"WARNING: Not enough fake articles for {language}")

    if len(real_files) < SAMPLES_PER_CLASS:
        print(f"WARNING: Not enough real articles for {language}")

    # Randomly select samples
    random.shuffle(fake_files)
    random.shuffle(real_files)

    fake_selected = fake_files[:SAMPLES_PER_CLASS]
    real_selected = real_files[:SAMPLES_PER_CLASS]

    # ==============================
    # PROCESS FAKE
    # ==============================

    for index, file_path in enumerate(fake_selected, start=1):

        title, body = read_article(file_path)

        text = f"{title} {body}".strip()

        if not text:
            continue

        record = {
            "id": f"{language_code}_FAKE_{index:05d}",
            "language": language,
            "title": title,
            "body": body,
            "text": text,
            "label": 0,
            "source_file": os.path.basename(file_path)
        }

        all_records.append(record)

    # ==============================
    # PROCESS REAL
    # ==============================

    for index, file_path in enumerate(real_selected, start=1):

        title, body = read_article(file_path)

        text = f"{title} {body}".strip()

        if not text:
            continue

        record = {
            "id": f"{language_code}_REAL_{index:05d}",
            "language": language,
            "title": title,
            "body": body,
            "text": text,
            "label": 1,
            "source_file": os.path.basename(file_path)
        }

        all_records.append(record)


# ==============================
# CREATE DATAFRAME
# ==============================

df = pd.DataFrame(all_records)

# Shuffle the final dataset
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Save
output_file = os.path.join(
    PROCESSED_DIR,
    "multilingual_fake_news.csv"
)

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ==============================
# SHOW RESULTS
# ==============================

print("\n")
print("=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(f"Total articles: {len(df)}")

print("\nLanguage distribution:")
print(df["language"].value_counts())

print("\nFake / Real distribution:")
print(df["label"].value_counts())

print("\nLanguage + Label distribution:")
print(
    df.groupby(["language", "label"])
      .size()
      .unstack(fill_value=0)
)

print("\nSaved dataset:")
print(output_file)

print("\nFirst 5 rows:")
print(
    df[
        ["id", "language", "title", "label"]
    ].head()
)