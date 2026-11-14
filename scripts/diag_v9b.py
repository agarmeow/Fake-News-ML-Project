import os
import json
import hashlib
import pandas as pd

def clean_v(v):
    if pd.isna(v):
        return "UNVERIFIABLE"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return str(v).strip().upper()

def get_cached_search_results(query):
    h = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
    path = os.path.join("cache", "search", f"{h}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("results", [])
    return None

# 1. Count cache/search entries with empty results
search_dir = os.path.join("cache", "search")
empty_cache_count = 0
total_cache_count = 0
if os.path.exists(search_dir):
    for f in os.listdir(search_dir):
        if f.endswith(".json"):
            total_cache_count += 1
            with open(os.path.join(search_dir, f), "r", encoding="utf-8") as fp:
                try:
                    d = json.load(fp)
                    res = d.get("results", [])
                    if not res:
                        empty_cache_count += 1
                except Exception:
                    pass

print(f"cache/search total entries: {total_cache_count}, empty result entries: {empty_cache_count}")

# 2. Check REPORT set
df_abl = pd.read_csv("results/verifier_ablation.csv")
exclude = [f"T{i:02d}" for i in range(1, 7)] + [f"F{i:02d}" for i in range(1, 7)]
df_rep = df_abl[~df_abl["id"].isin(exclude)]

zero_abl = []
for _, r in df_rep.iterrows():
    q = r["claim_n"]
    res = get_cached_search_results(q)
    if res is not None and len(res) == 0:
        zero_abl.append((r["id"], clean_v(r["gold"]), q))

print(f"\nAblation (REPORT set) rows with 0 evidence passages (N={len(zero_abl)}):")
for zid, zgold, zq in zero_abl:
    print(f"  id={zid}, gold={zgold}, claim='{zq}'")

# 3. Check relation_claims_pattern.csv
df_rel = pd.read_csv("results/verifier_relation_pattern.csv")
zero_rel = []
for _, r in df_rel.iterrows():
    q = r["claim_n"]
    res = get_cached_search_results(q)
    if res is not None and len(res) == 0:
        zero_rel.append((r["id"], clean_v(r["gold"]), q))

print(f"\nRelation pattern rows with 0 evidence passages (N={len(zero_rel)}):")
for zid, zgold, zq in zero_rel:
    print(f"  id={zid}, gold={zgold}, claim='{zq}'")
