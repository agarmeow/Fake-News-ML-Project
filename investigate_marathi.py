import os
import re
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXTRACTED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "extracted",
    "marathi"
)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


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

    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    if not lines:
        return "", ""

    title = lines[0]
    body = " ".join(lines[1:]).strip()

    if body.startswith(title):
        body = body[len(title):].strip()

    return title, body


# ============================================================
# FIND MARATHI REAL FILES
# ============================================================

real_files = []

for root, dirs, files in os.walk(EXTRACTED_DIR):

    dirs[:] = [
        d for d in dirs
        if d != "__MACOSX"
    ]

    for filename in files:

        if not filename.lower().endswith(".txt"):
            continue

        if "_real_" in filename.lower():
            real_files.append(
                os.path.join(root, filename)
            )


print("=" * 60)
print("MARATHI REAL DATASET INVESTIGATION")
print("=" * 60)

print("\nTotal Marathi Real files:")
print(len(real_files))


# ============================================================
# READ ALL ARTICLES
# ============================================================

records = []

for file_path in real_files:

    title, body = read_article(file_path)

    text = f"{title} {body}".strip()

    records.append({
        "filename": os.path.basename(file_path),
        "title": title,
        "body": body,
        "text": text
    })


df = pd.DataFrame(records)


# ============================================================
# UNIQUE COUNTS
# ============================================================

print("\n" + "=" * 60)
print("UNIQUENESS ANALYSIS")
print("=" * 60)

print("\nTotal articles:")
print(len(df))

print("\nUnique titles:")
print(
    df["title"]
    .map(normalize_text)
    .nunique()
)

print("\nUnique bodies:")
print(
    df["body"]
    .map(normalize_text)
    .nunique()
)

print("\nUnique complete texts:")
print(
    df["text"]
    .map(normalize_text)
    .nunique()
)


# ============================================================
# TITLE DUPLICATION
# ============================================================

title_counts = (
    df["title"]
    .map(normalize_text)
    .value_counts()
)

print("\n" + "=" * 60)
print("MOST REPEATED TITLES")
print("=" * 60)

print(
    title_counts.head(10).to_string()
)


# ============================================================
# SHOW A FEW REPEATED ARTICLES
# ============================================================

print("\n" + "=" * 60)
print("REPEATED ARTICLE EXAMPLES")
print("=" * 60)

text_counts = (
    df["text"]
    .map(normalize_text)
    .value_counts()
)

repeated_texts = text_counts[
    text_counts > 1
]

print(
    "\nNumber of texts appearing more than once:",
    len(repeated_texts)
)

shown = 0

for text, count in repeated_texts.items():

    group = df[
        df["text"].map(normalize_text) == text
    ]

    print("\n----------------------------------------")
    print("Occurrences:", count)

    print("\nFiles:")

    for filename in group["filename"].head(10):
        print(" ", filename)

    print("\nTitle:")
    print(group.iloc[0]["title"])

    print("\nBody preview:")
    print(
        str(group.iloc[0]["body"])[:400]
    )

    shown += 1

    if shown >= 5:
        break


# ============================================================
# ARTICLE LENGTH
# ============================================================

df["text_length"] = (
    df["text"]
    .fillna("")
    .str.len()
)

print("\n" + "=" * 60)
print("ARTICLE LENGTH")
print("=" * 60)

print(
    df["text_length"].describe()
)


# ============================================================
# SAVE INVESTIGATION DATA
# ============================================================

output_file = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "marathi_real_investigation.csv"
)

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nInvestigation data saved to:")
print(output_file)

print("\n" + "=" * 60)
print("INVESTIGATION COMPLETE")
print("=" * 60)