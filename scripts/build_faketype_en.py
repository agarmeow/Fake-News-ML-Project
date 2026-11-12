import os
import re
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from reasoning.text_prep import prep_text

DATA_DIR = PROJECT_ROOT / "data" / "faketype_en"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Collect all texts from data/eval/*.csv for exclusion
eval_texts = set()
for eval_path in EVAL_DIR.glob("*.csv"):
    try:
        edf = pd.read_csv(eval_path)
        for col in ["text", "claim", "title", "headline"]:
            if col in edf.columns:
                for t in edf[col].dropna().astype(str):
                    c = prep_text(t)
                    if c:
                        eval_texts.add(c)
    except Exception as e:
        print(f"Warning reading {eval_path}: {e}")
print(f"Loaded {len(eval_texts)} eval texts to exclude.")

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

# 3. Source casing stats (computed BEFORE prep_text on raw sources)
source_stats = {}
sources_dict = {
    "Onion": onion_raw,
    "HuffPost": huffpost_raw,
    "Chakraborty-clickbait": clickbait_raw,
    "Chakraborty-nonclickbait": non_cb_raw,
}

for s_name, s_texts in sources_dict.items():
    n_tot = len(s_texts)
    share_upper = sum(bool(re.search(r"[A-Z]", s)) for s in s_texts) / n_tot if n_tot > 0 else 0.0
    share_title = sum(s.istitle() for s in s_texts) / n_tot if n_tot > 0 else 0.0
    source_stats[s_name] = {"share_uppercase": share_upper, "share_titlecase": share_title}


def filter_texts(texts):
    unique_filtered = []
    seen = set()
    for t in texts:
        clean_s = re.sub(r"\s+", " ", t).strip()
        if len(clean_s.split()) < 3:
            continue
        p = prep_text(clean_s)
        if p in seen or p in eval_texts:
            continue
        seen.add(p)
        unique_filtered.append(clean_s)
    return unique_filtered


onion_clean = filter_texts(onion_raw)
huffpost_clean = filter_texts(huffpost_raw)
clickbait_clean = filter_texts(clickbait_raw)
non_cb_clean = filter_texts(non_cb_raw)

# 4. Assemble classes (cap 8000 per class, seed 42)
rng = np.random.RandomState(42)

satire_sampled = list(rng.choice(onion_clean, size=8000, replace=False))
clickbait_sampled = list(rng.choice(clickbait_clean, size=8000, replace=False))

huff_sampled = list(rng.choice(huffpost_clean, size=4000, replace=False))
non_cb_sampled = list(rng.choice(non_cb_clean, size=4000, replace=False))
plain_sampled = huff_sampled + non_cb_sampled
rng.shuffle(plain_sampled)

# 5. Digit stats per class (computed BEFORE prep_text)
class_raw_dict = {
    "plain": plain_sampled,
    "satire": satire_sampled,
    "clickbait": clickbait_sampled,
}

class_digit_stats = {}
for c_name, c_texts in class_raw_dict.items():
    n_tot = len(c_texts)
    share_digit = sum(bool(re.search(r"\d", s)) for s in c_texts) / n_tot if n_tot > 0 else 0.0
    share_start_digit = sum(bool(re.match(r"^\d", s.strip())) for s in c_texts) / n_tot if n_tot > 0 else 0.0
    class_digit_stats[c_name] = {"share_has_digit": share_digit, "share_starts_digit": share_start_digit}

# 6. Apply prep_text AFTER computing stats
df_satire = pd.DataFrame({"text": [prep_text(s) for s in satire_sampled], "label": "satire"})
df_clickbait = pd.DataFrame({"text": [prep_text(s) for s in clickbait_sampled], "label": "clickbait"})
df_plain = pd.DataFrame({"text": [prep_text(s) for s in plain_sampled], "label": "plain"})

full_df = pd.concat([df_plain, df_satire, df_clickbait], ignore_index=True)

# 7. Stratified 80/10/10 split
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

# 8. Mean word length per class (on prepped text)
mean_word_lengths = {}
for lbl in ["plain", "satire", "clickbait"]:
    sub = full_df[full_df["label"] == lbl]["text"]
    lengths = sub.apply(lambda s: len(s.split()))
    mean_word_lengths[lbl] = lengths.mean()

# 9. Top 20 TF-IDF+LR features per class
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

# 10. Write results/faketype_en_audit.txt
audit_path = RESULTS_DIR / "faketype_en_audit.txt"
with open(audit_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("FAKETYPE ENGLISH DATASET AUDIT (A2b REBUILD)\n")
    f.write("=" * 60 + "\n\n")

    f.write("PER-SOURCE CASING STATS (BEFORE PREP_TEXT):\n")
    f.write("Source                     Any Uppercase   Title Case\n")
    f.write("-" * 55 + "\n")
    for s_name, st in source_stats.items():
        f.write(f"{s_name:<26} {st['share_uppercase']:<15.4f} {st['share_titlecase']:<10.4f}\n")

    f.write("\nPER-CLASS DIGIT STATS (BEFORE PREP_TEXT):\n")
    f.write("Class        Contains Digit   Starts with Digit\n")
    f.write("-" * 50 + "\n")
    for c_name, st in class_digit_stats.items():
        f.write(f"{c_name:<12} {st['share_has_digit']:<16.4f} {st['share_starts_digit']:<15.4f}\n")

    f.write("\nPER-CLASS COUNTS PER SPLIT:\n")
    f.write("Class        Train     Val    Test    Total\n")
    f.write("-" * 45 + "\n")
    for cls in ["plain", "satire", "clickbait"]:
        tr_c = (train_df["label"] == cls).sum()
        v_c = (val_df["label"] == cls).sum()
        te_c = (test_df["label"] == cls).sum()
        tot = (full_df["label"] == cls).sum()
        f.write(f"{cls:<12} {tr_c:<9} {v_c:<6} {te_c:<7} {tot:<7}\n")

    f.write("\nMEAN WORD LENGTH PER CLASS (PREPPED):\n")
    for cls in ["plain", "satire", "clickbait"]:
        f.write(f"  {cls:<12}: {mean_word_lengths[cls]:.2f} words\n")

    f.write("\nTOP 20 TF-IDF+LR FEATURES PER CLASS:\n")
    for cls in ["plain", "satire", "clickbait"]:
        f.write(f"\n[{cls}]:\n")
        f.write(", ".join(top20_per_class[cls]) + "\n")

print(f"Audit written successfully to {audit_path}")
