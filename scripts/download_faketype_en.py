import os
import gzip
import json
from pathlib import Path
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "faketype_en"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"

# 1. Confirm data/ is gitignored
with open(GITIGNORE_PATH, "r", encoding="utf-8") as f:
    gitignore_content = f.read()
assert "data/" in gitignore_content, "data/ is not gitignored!"
print("Confirmed: data/ is gitignored.")

DATA_DIR.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}

# (a) Misra Onion/HuffPost headlines
sarcasm_url = "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json"
sarcasm_dest = DATA_DIR / "Sarcasm_Headlines_Dataset_v2.json"
print(f"Downloading Misra Sarcasm headlines from {sarcasm_url}...")
resp_sarcasm = requests.get(sarcasm_url, headers=headers, verify=False, timeout=30)
with open(sarcasm_dest, "wb") as f:
    f.write(resp_sarcasm.content)

# Count sarcasm rows
with open(sarcasm_dest, "r", encoding="utf-8") as f:
    sarcasm_rows = len([line for line in f if line.strip()])
print(f"File: {sarcasm_dest.name} | Row count: {sarcasm_rows}")

# (b) Chakraborty clickbait headlines
cb_url = "https://raw.githubusercontent.com/bhargaviparanjape/clickbait/master/dataset/clickbait_data.gz"
non_cb_url = "https://raw.githubusercontent.com/bhargaviparanjape/clickbait/master/dataset/non_clickbait_data.gz"

cb_dest = DATA_DIR / "clickbait_data.txt"
non_cb_dest = DATA_DIR / "non_clickbait_data.txt"

print(f"Downloading Chakraborty clickbait from {cb_url}...")
resp_cb = requests.get(cb_url, headers=headers, verify=False, timeout=30)
cb_decomp = gzip.decompress(resp_cb.content)
with open(cb_dest, "wb") as f:
    f.write(cb_decomp)

with open(cb_dest, "r", encoding="utf-8") as f:
    cb_rows = len([line for line in f if line.strip()])
print(f"File: {cb_dest.name} | Row count: {cb_rows}")

print(f"Downloading Chakraborty non-clickbait from {non_cb_url}...")
resp_non_cb = requests.get(non_cb_url, headers=headers, verify=False, timeout=30)
non_cb_decomp = gzip.decompress(resp_non_cb.content)
with open(non_cb_dest, "wb") as f:
    f.write(non_cb_decomp)

with open(non_cb_dest, "r", encoding="utf-8") as f:
    non_cb_rows = len([line for line in f if line.strip()])
print(f"File: {non_cb_dest.name} | Row count: {non_cb_rows}")
