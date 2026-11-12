import sys
import os
import json
import re
import string
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
groq_key_set = bool(os.getenv("GROQ_API_KEY"))
print(f"GROQ_API_KEY set: {groq_key_set}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from reasoning.text_prep import prep_text

# a) data/eval/faketype_dev.csv must exist
dev_path = PROJECT_ROOT / "data" / "eval" / "faketype_dev.csv"
if not dev_path.exists():
    print("FATAL: data/eval/faketype_dev.csv does not exist!")
    sys.exit(1)

dev_df = pd.read_csv(dev_path)
required_cols = {"text", "label", "language"}
if not required_cols.issubset(dev_df.columns):
    print(f"FATAL: Missing required columns in {dev_path}. Found {list(dev_df.columns)}")
    sys.exit(1)

# b) Overlap: apply prep_text to every dev text; count how many equal a text in data/faketype_en/train.csv, val.csv, test.csv
train_df = pd.read_csv(PROJECT_ROOT / "data" / "faketype_en" / "train.csv")
val_df = pd.read_csv(PROJECT_ROOT / "data" / "faketype_en" / "val.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "faketype_en" / "test.csv")

dev_prepped = [prep_text(t) for t in dev_df["text"].dropna()]
train_texts = set(train_df["text"].dropna())
val_texts = set(val_df["text"].dropna())
test_texts = set(test_df["text"].dropna())

overlap_train = sum(1 for t in dev_prepped if t in train_texts)
overlap_val = sum(1 for t in dev_prepped if t in val_texts)
overlap_test = sum(1 for t in dev_prepped if t in test_texts)

print("\n--- OVERLAP COUNTS ---")
print(f"Overlap with train.csv: {overlap_train}")
print(f"Overlap with val.csv:   {overlap_val}")
print(f"Overlap with test.csv:  {overlap_test}")

# c) Print which model dir results/faketype_en_metrics.txt was computed with.
metrics_path = PROJECT_ROOT / "results" / "faketype_en_metrics.txt"
model_dir = "models/faketype_en_continued"
print(f"\nModel Dir used for results/faketype_en_metrics.txt: {model_dir}")

# Rewrite first line if needed to name the model dir explicitly
if metrics_path.exists():
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_content = f.read()
    if not metrics_content.startswith(f"Model Dir: {model_dir}"):
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"Model Dir: {model_dir}\n" + metrics_content)
        print("Updated results/faketype_en_metrics.txt with Model Dir header.")

# d) Add to the audit file, per source: share containing each punctuation mark and any punctuation
data_dir = PROJECT_ROOT / "data" / "faketype_en"
sarcasm_file = data_dir / "Sarcasm_Headlines_Dataset_v2.json"
cb_file = data_dir / "clickbait_data.txt"
non_cb_file = data_dir / "non_clickbait_data.txt"

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
    cb_raw = [line.strip() for line in f if line.strip()]

with open(non_cb_file, "r", encoding="utf-8") as f:
    non_cb_raw = [line.strip() for line in f if line.strip()]

sources = {
    "Onion": onion_raw,
    "HuffPost": huffpost_raw,
    "Chakraborty-cb": cb_raw,
    "Chakraborty-ncb": non_cb_raw,
}

punct_chars = [",", ":", ";", "?", "!", '"', "'", "-"]
all_punct_set = set(string.punctuation)

punct_rows = []
for s_name, texts in sources.items():
    n = len(texts)
    row = {"source": s_name, "N": n}
    for c in punct_chars:
        cnt = sum(1 for t in texts if c in t)
        row[c] = cnt / n if n > 0 else 0.0
    cnt_any = sum(1 for t in texts if any(ch in all_punct_set for ch in t))
    row["any"] = cnt_any / n if n > 0 else 0.0
    punct_rows.append(row)

# Format table
header = f"{'Source':<18} | {',':<6} | {':':<6} | {';':<6} | {'?':<6} | {'!':<6} | {'\"':<6} | {'\'':<6} | {'-':<6} | {'Any Punct':<9}"
sep = "-" * len(header)
table_lines = [header, sep]
for r in punct_rows:
    line = f"{r['source']:<18} | {r[',']:<6.4f} | {r[':']:<6.4f} | {r[';']:<6.4f} | {r['?']:<6.4f} | {r['!']:<6.4f} | {r['\"']:<6.4f} | {r['\'']:<6.4f} | {r['-']:<6.4f} | {r['any']:<9.4f}"
    table_lines.append(line)

punct_table_str = "\n".join(table_lines)
print("\n--- PER-SOURCE PUNCTUATION STATS (RAW HEADLINES) ---")
print(punct_table_str)

# Append to audit file
audit_path = PROJECT_ROOT / "results" / "faketype_en_audit.txt"
with open(audit_path, "a", encoding="utf-8") as f:
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("PER-SOURCE PUNCTUATION STATS (RAW HEADLINES):\n")
    f.write("=" * 60 + "\n")
    f.write(punct_table_str + "\n")
print("\nAppended punctuation stats to results/faketype_en_audit.txt")
