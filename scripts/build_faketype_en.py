import os
import re
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "faketype_en"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Collect all texts from data/eval/*.csv for deduplication
eval_texts = set()
for eval_path in EVAL_DIR.glob("*.csv"):
    try:
        edf = pd.read_csv(eval_path)
        for col in ["text", "claim", "title", "headline"]:
            if col in edf.columns:
                for t in edf[col].dropna().astype(str):
                    cleaned = re.sub(r"\s+", " ", t).strip().lower()
                    if cleaned:
                        eval_texts.add(cleaned)
    except Exception as e:
        print(f"Warning reading {eval_path}: {e}")
print(f"Loaded {len(eval_texts)} eval texts to exclude.")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    return t


def filter_texts(texts):
    unique_filtered = []
    seen = set()
    for t in texts:
        c = clean_text(t)
        c_lower = c.lower()
        # Drop under 3 words
        if len(c.split()) < 3:
            continue
        # Drop duplicates
        if c_lower in seen:
            continue
        # Drop if in eval
        if c_lower in eval_texts:
            continue
        seen.add(c_lower)
        unique_filtered.append(c)
    return unique_filtered


# 2. Load raw data
sarcasm_file = DATA_DIR / "Sarcasm_Headlines_Dataset_v2.json"
cb_file = DATA_DIR / "clickbait_data.txt"
non_cb_file = DATA_DIR / "non_clickbait_data.txt"

onion_raw = []
huffpost_raw = []

with open(sarcasm_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        h = item.get("headline", "")
        if item.get("is_sarcastic") == 1:
            onion_raw.append(h)
        else:
            huffpost_raw.append(h)

with open(cb_file, "r", encoding="utf-8") as f:
    clickbait_raw = [line.strip() for line in f if line.strip()]

with open(non_cb_file, "r", encoding="utf-8") as f:
    non_cb_raw = [line.strip() for line in f if line.strip()]

# 3. Clean and filter
onion_clean = filter_texts(onion_raw)
huffpost_clean = filter_texts(huffpost_raw)
clickbait_clean = filter_texts(clickbait_raw)
non_cb_clean = filter_texts(non_cb_raw)

print(f"Filtered pool counts:")
print(f"  Onion (satire): {len(onion_clean)}")
print(f"  HuffPost (plain part 1): {len(huffpost_clean)}")
print(f"  Chakraborty non-clickbait (plain part 2): {len(non_cb_clean)}")
print(f"  Chakraborty clickbait: {len(clickbait_clean)}")

# 4. Assemble classes (cap 8000 per class, seed 42)
rng = np.random.RandomState(42)

# Satire: 8000 from Onion
satire_sampled = list(rng.choice(onion_clean, size=8000, replace=False))

# Clickbait: 8000 from Chakraborty clickbait
clickbait_sampled = list(rng.choice(clickbait_clean, size=8000, replace=False))

# Plain: half HuffPost (4000) + half Chakraborty non-clickbait (4000)
huff_sampled = list(rng.choice(huffpost_clean, size=4000, replace=False))
non_cb_sampled = list(rng.choice(non_cb_clean, size=4000, replace=False))
plain_sampled = huff_sampled + non_cb_sampled
rng.shuffle(plain_sampled)

# Combine into DataFrame
df_satire = pd.DataFrame({"text": satire_sampled, "label": "satire"})
df_clickbait = pd.DataFrame({"text": clickbait_sampled, "label": "clickbait"})
df_plain = pd.DataFrame({"text": plain_sampled, "label": "plain"})

full_df = pd.concat([df_plain, df_satire, df_clickbait], ignore_index=True)

# 5. Stratified 80/10/10 split
train_df, temp_df = train_test_split(
    full_df, test_size=0.20, stratify=full_df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

# Save splits
train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)
full_df.to_csv(DATA_DIR / "faketype_en_all.csv", index=False)

print("\nSplit counts:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")

# 6. Mean word length per class (over full_df)
mean_word_lengths = {}
for lbl in ["plain", "satire", "clickbait"]:
    sub = full_df[full_df["label"] == lbl]["text"]
    lengths = sub.apply(lambda s: len(s.split()))
    mean_word_lengths[lbl] = lengths.mean()

# 7. Top 20 TF-IDF+LR features per class
tfidf = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
X_train = tfidf.fit_transform(train_df["text"])
y_train = train_df["label"].values

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

feature_names = np.array(tfidf.get_feature_names_out())
classes = lr.classes_

top20_per_class = {}
for i, cls in enumerate(classes):
    coefs = lr.coef_[i]
    top_indices = np.argsort(coefs)[::-1][:20]
    top20_per_class[cls] = feature_names[top_indices]

# 8. Write results/faketype_en_audit.txt
audit_path = RESULTS_DIR / "faketype_en_audit.txt"
with open(audit_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("FAKETYPE ENGLISH DATASET AUDIT & TOP FEATURES\n")
    f.write("=" * 60 + "\n\n")

    f.write("PER-CLASS COUNTS PER SPLIT:\n")
    f.write("Class        Train     Val    Test    Total\n")
    f.write("-" * 45 + "\n")
    for cls in ["plain", "satire", "clickbait"]:
        tr_c = (train_df["label"] == cls).sum()
        v_c = (val_df["label"] == cls).sum()
        te_c = (test_df["label"] == cls).sum()
        tot = (full_df["label"] == cls).sum()
        f.write(f"{cls:<12} {tr_c:<9} {v_c:<6} {te_c:<7} {tot:<7}\n")

    f.write("\nMEAN WORD LENGTH PER CLASS:\n")
    for cls in ["plain", "satire", "clickbait"]:
        f.write(f"  {cls:<12}: {mean_word_lengths[cls]:.2f} words\n")

    f.write("\nTOP 20 TF-IDF+LR FEATURES PER CLASS:\n")
    for cls in ["plain", "satire", "clickbait"]:
        f.write(f"\n[{cls}]:\n")
        f.write(", ".join(top20_per_class[cls]) + "\n")

print(f"\nAudit successfully written to {audit_path}")
