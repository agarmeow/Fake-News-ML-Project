import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from verification.consistency_checker import extract_relation

claims_csv = PROJECT_ROOT / "data" / "eval" / "relation_claims.csv"
df = pd.read_csv(claims_csv)

print("--- ORIGINAL CLAIMS (data/eval/relation_claims.csv) ---")
for idx, r in df.iterrows():
    c = str(r["claim"]).strip()
    rel = extract_relation(c)
    print(f"{r['id']}: '{c}' -> {rel}")

# Phrasing: "<Place>'s capital is <City>."
# e.g. "The capital of Australia is Canberra." -> "Australia's capital is Canberra."
pattern_records = []
print("\n--- PATTERN CLAIMS (<Place>'s capital is <City>.) ---")
for idx, r in df.iterrows():
    c = str(r["claim"]).strip()
    # Match "The capital of <Place> is <City>."
    m = re.match(r"^The capital of\s+(.+?)\s+is\s+(.+?)\.?$", c, re.IGNORECASE)
    if m:
        place = m.group(1).strip()
        city = m.group(2).strip().rstrip(".")
        new_claim = f"{place}'s capital is {city}."
    else:
        new_claim = c

    rel = extract_relation(new_claim)
    print(f"{r['id']}p: '{new_claim}' -> {rel}")
    pattern_records.append({
        "id": f"{r['id']}p",
        "claim": new_claim,
        "label": r["label"],
        "category": "relation_pattern",
        "language": r.get("language", "en"),
        "note": r.get("note", ""),
    })

pattern_df = pd.DataFrame(pattern_records)
out_csv = PROJECT_ROOT / "data" / "eval" / "relation_claims_pattern.csv"
pattern_df.to_csv(out_csv, index=False)
print(f"\nWrote {len(pattern_df)} rows to {out_csv}.")
