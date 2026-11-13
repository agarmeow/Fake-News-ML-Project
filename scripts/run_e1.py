import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from dotenv import load_dotenv

# HuggingFace SSL workaround
import httpx
from huggingface_hub.utils import _http
_http.set_client_factory(lambda: httpx.Client(verify=False, follow_redirects=True, timeout=60.0))

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY set: {bool(groq_key)}")
if not groq_key:
    print("FATAL: GROQ_API_KEY is not set!")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fact_check.fact_checker import fact_check

# Load existing dev and test results to get v1 final predictions and trained/LLM labels
results_dir = PROJECT_ROOT / "results"
dev_res_df = pd.read_csv(results_dir / "faketype_en_dev.csv") if (results_dir / "faketype_en_dev.csv").exists() else pd.DataFrame()
test_res_df = pd.read_csv(results_dir / "faketype_en_test.csv") if (results_dir / "faketype_en_test.csv").exists() else pd.DataFrame()

v1_lookup = {}
for df_src in [dev_res_df, test_res_df]:
    if not df_src.empty:
        for _, r in df_src.iterrows():
            t = str(r["text"]).strip()
            v1_lookup[t] = {
                "pred_pipeline": r.get("pred_pipeline", ""),
                "trained": f"{r.get('pred_trained', '')} ({r.get('conf_trained', 0.0):.3f})",
                "llm": f"{r.get('pred_llm1', '')} ({r.get('conf_llm1', 0.0):.3f})",
            }

out_path = results_dir / "faketype_en_verdicts.csv"
existing_verdicts = {}
if out_path.exists():
    try:
        prev_df = pd.read_csv(out_path)
        for _, r in prev_df.iterrows():
            if str(r.get("verdict_source")) != "error":
                existing_verdicts[str(r["text"]).strip()] = r.to_dict()
        print(f"Resuming: found {len(existing_verdicts)} existing verdicts.")
    except Exception as e:
        print(f"Warning reading existing verdicts: {e}")

splits = [
    ("dev", PROJECT_ROOT / "data" / "eval" / "faketype_dev.csv"),
    ("test", PROJECT_ROOT / "data" / "eval" / "faketype_test.csv"),
]

all_rows = []
for split_name, csv_path in splits:
    df = pd.read_csv(csv_path)
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == "en"].copy()

    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        gold = str(row["label"]).strip().lower()
        if gold in ("ordinary news", "plain", "factual", "normal"):
            gold = "ordinary"

        if text in existing_verdicts:
            all_rows.append(existing_verdicts[text])
            continue

        print(f"[{len(all_rows) + 1}] Fact-checking ({split_name}): {text[:60]}...")
        try:
            fc_res = fact_check(text)
            verdict = str(fc_res.get("verdict", "UNVERIFIABLE")).upper()
            verdict_source = str(fc_res.get("verdict_source", "none"))
        except Exception as e:
            print(f"Error fact checking '{text[:40]}...': {e}")
            verdict = "UNVERIFIABLE"
            verdict_source = "error"

        row_dict = {
            "split": split_name,
            "text": text,
            "gold": gold,
            "verdict": verdict,
            "verdict_source": verdict_source,
        }
        all_rows.append(row_dict)
        existing_verdicts[text] = row_dict
        pd.DataFrame(all_rows).to_csv(out_path, index=False)

verdicts_df = pd.DataFrame(all_rows)
verdicts_df.to_csv(out_path, index=False)
print(f"\nWritten {len(verdicts_df)} verdicts to {out_path}.")

# (a) Verdict counts (TRUE/FALSE/UNVERIFIABLE) per gold class
print("\n" + "=" * 60)
print("(A) VERDICT COUNTS PER GOLD CLASS (DEV + TEST, N=76):")
print("=" * 60)
classes = ["ordinary", "satire", "clickbait", "misleading"]
summary_rows = []
for c in classes:
    sub = verdicts_df[verdicts_df["gold"] == c]
    n_tot = len(sub)
    n_true = (sub["verdict"] == "TRUE").sum()
    n_false = (sub["verdict"] == "FALSE").sum()
    n_unver = (sub["verdict"] == "UNVERIFIABLE").sum()
    summary_rows.append({
        "Gold Class": c,
        "Total": n_tot,
        "TRUE": n_true,
        "FALSE": n_false,
        "UNVERIFIABLE": n_unver,
    })

sum_df = pd.DataFrame(summary_rows)
print(sum_df.to_string(index=False))

# (b) For rows with gold == misleading and v1 final == satire: trained label+conf, LLM label+conf, verdict
print("\n" + "=" * 60)
print("(B) ROWS WITH GOLD == MISLEADING AND V1 FINAL == SATIRE:")
print("=" * 60)
misl_satire_rows = []
for _, r in verdicts_df[verdicts_df["gold"] == "misleading"].iterrows():
    t = r["text"]
    v1_info = v1_lookup.get(t, {})
    if v1_info.get("pred_pipeline") == "satire":
        misl_satire_rows.append({
            "split": r["split"],
            "text": t,
            "trained": v1_info.get("trained", ""),
            "llm": v1_info.get("llm", ""),
            "verdict": r["verdict"],
        })

print(f"Found {len(misl_satire_rows)} rows matching condition:")
for i, item in enumerate(misl_satire_rows, 1):
    print(f"\n[{i}] ({item['split']}) \"{item['text']}\"")
    print(f"    Trained: {item['trained']} | LLM: {item['llm']} | Verdict: {item['verdict']}")
