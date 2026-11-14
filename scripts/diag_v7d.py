import sys
import re
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "src")

from fact_check.fact_checker import fact_check

claim = "Europe's Hera spacecraft has begun close inspections of an asteroid rammed by NASA in a 2022 planetary defense test."

res = fact_check(claim)

print(f"Claim: {claim}")
print(f"Final Verdict: {res['verdict']} (Confidence: {res['confidence']:.4f}, Source: {res['verdict_source']})")
print(f"Support score: {res['support_score']:.4f}, Contradiction score: {res['contradiction_score']:.4f}")

# Inspect evidence
evidence_items = res["evidence"]

def extract_pub_date(url, cached_text):
    # Try finding date in URL first
    m_url = re.search(r"/(20\d{2})[-/](0[1-9]|1[0-2])[-/]([0-3]\d)/", url)
    if m_url:
        return f"{m_url.group(1)}-{m_url.group(2)}-{m_url.group(3)}"
    
    # Check text for publication date patterns
    if not cached_text:
        return "n/a"
    
    patterns = [
        r"(?:published|updated|date):\s*([a-zA-Z]+ \d{1,2}, \d{4})",
        r"(?:published|updated|date):\s*(\d{4}-\d{2}-\d{2})",
        r"([a-zA-Z]+ \d{1,2}, \d{4})",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for p in patterns:
        m = re.search(p, cached_text, re.IGNORECASE)
        if m:
            return m.group(1)
    return "n/a"

print("\nPER-EVIDENCE TABLE:")
header = f"{'Title[:40]':<40} | {'Pub Date':<15} | {'NLI Label':<12} | {'NLI Conf':<8} | {'Consistency':<12} | {'Counted As':<10}"
print(header)
print("-" * len(header))

for item in evidence_items:
    title = str(item.get("title", ""))[:40]
    url = item.get("url", "")
    text = item.get("evidence", "")
    
    # Try finding cached file in cache/pages
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    cached_file = os.path.join("cache", "pages", f"{url_hash}.json")
    full_text = text
    if os.path.exists(cached_file):
        try:
            import json
            with open(cached_file, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                full_text = cdata.get("text", text)
        except Exception:
            pass

    pub_date = extract_pub_date(url, full_text)
    nli_label = str(item.get("nli_label", "n/a"))
    nli_conf = f"{float(item.get('nli_confidence', 0.0)):.4f}"
    cons = str(item.get("consistency_verdict", "n/a"))
    counted = str(item.get("counted_as", "discarded"))
    
    print(f"{title:<40} | {pub_date:<15} | {nli_label:<12} | {nli_conf:<8} | {cons:<12} | {counted:<10}")

print("\nEvidence text samples:")
for i, item in enumerate(evidence_items, 1):
    print(f"[{i}] {item.get('title')}: {item.get('evidence')[:200]}...")
