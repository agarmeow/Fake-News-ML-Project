import os
import sys

# Force UTF-8 encoding for stdout and stderr on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import time
import csv
import pandas as pd
from collections import Counter

# Set repo root in path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from reasoning.claim_normalizer import normalize_claim
from fact_check.fact_checker import fact_check
from reasoning.type_view import get_type_view, is_english

OUT_CSV = os.path.join(REPO_ROOT, "results", "display_contract.csv")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

VALID_TYPES = {"ordinary", "satire", "clickbait", "misleading"}
VALID_VERDICTS = {"TRUE", "FALSE", "UNVERIFIABLE"}

FILES_CONFIG = [
    ("data/eval/claims.csv", "claim", "id"),
    ("data/eval/faketype_dev.csv", "text", None),
    ("data/eval/faketype_holdout.csv", "text", None),
]

def load_resumed_records():
    if not os.path.exists(OUT_CSV):
        return {}
    done = {}
    try:
        df = pd.read_csv(OUT_CSV)
        for _, r in df.iterrows():
            key = (str(r.get("file")), str(r.get("row_id")))
            # Only consider done if no exception occurred
            if pd.isna(r.get("exception")):
                done[key] = r.to_dict()
    except Exception as e:
        print(f"Warning loading existing CSV: {e}")
    return done

def main():
    done_records = load_resumed_records()
    print(f"Found {len(done_records)} clean pre-existing records in {OUT_CSV}")

    fieldnames = [
        "file", "row_id", "text", "language", "is_english",
        "verdict", "verdict_source", "no_evidence", "valid_verdict",
        "type_shown", "type_label", "type_source", "type_error", "valid_type",
        "exception"
    ]

    all_records_map = dict(done_records)
    consecutive_groq_failures = 0

    for file_rel, text_col, id_col in FILES_CONFIG:
        file_path = os.path.join(REPO_ROOT, file_rel)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        print(f"\nProcessing {file_rel} ({len(df)} rows)...")

        for idx, row in df.iterrows():
            row_id = str(row[id_col]) if id_col and id_col in row and pd.notna(row[id_col]) else f"{os.path.basename(file_rel)}_{idx}"
            raw_text = str(row[text_col]) if text_col in row and pd.notna(row[text_col]) else ""
            key = (file_rel, row_id)

            if key in all_records_map:
                consecutive_groq_failures = 0
                continue

            rec = {
                "file": file_rel,
                "row_id": row_id,
                "text": raw_text,
                "language": None,
                "is_english": None,
                "verdict": None,
                "verdict_source": None,
                "no_evidence": False,
                "valid_verdict": False,
                "type_shown": False,
                "type_label": None,
                "type_source": None,
                "type_error": None,
                "valid_type": False,
                "exception": None
            }

            try:
                # Step 1: normalize_claim
                norm = normalize_claim(raw_text)
                norm_claim = norm.get("claim", raw_text)
                lang = norm.get("language")
                rec["language"] = lang
                is_eng = is_english(lang, norm_claim)
                rec["is_english"] = is_eng

                # Step 2: fact_check once
                v_res = fact_check(norm_claim)
                if not v_res:
                    v_res = {}
                verdict = v_res.get("verdict")
                no_ev = not v_res.get("evidence")
                rec["verdict"] = verdict
                rec["verdict_source"] = v_res.get("verdict_source")
                rec["no_evidence"] = no_ev

                valid_v = (verdict in VALID_VERDICTS) or (verdict is None and no_ev)
                rec["valid_verdict"] = valid_v

                # Check Groq status
                groq_failed = False
                llm_status = v_res.get("llm_status")

                # Step 3: get_type_view
                t_res = get_type_view(norm_claim, v_res, lang)
                type_shown = t_res.get("shown", False)
                type_label = t_res.get("label")
                type_err = t_res.get("error")
                rec["type_shown"] = type_shown
                rec["type_label"] = type_label
                rec["type_source"] = t_res.get("source")
                rec["type_error"] = type_err

                if type_err and ("rate" in str(type_err).lower() or "groq" in str(type_err).lower() or "429" in str(type_err)):
                    groq_failed = True
                elif llm_status in ("rate_limited", "error") and type_shown is False:
                    groq_failed = True

                # Check type contract
                if not is_eng:
                    valid_t = (type_shown is False) and (type_label is None)
                else:
                    valid_t = (type_shown is True and type_label in VALID_TYPES) or (type_shown is False and bool(type_err))
                rec["valid_type"] = valid_t

                # Guard rule: 5 consecutive Groq failures
                if groq_failed:
                    consecutive_groq_failures += 1
                    print(f"Warning: Groq failure on {row_id} (count={consecutive_groq_failures}), backing off 8s...")
                    time.sleep(8.0)
                    if consecutive_groq_failures >= 5:
                        print(f"ABORT: 5 consecutive Groq failures reached at {row_id}!")
                        all_records_map[key] = rec
                        pd.DataFrame(list(all_records_map.values())).to_csv(OUT_CSV, index=False, encoding="utf-8")
                        sys.exit(1)
                else:
                    consecutive_groq_failures = 0

            except Exception as e:
                rec["exception"] = str(e)
                print(f"Exception on {row_id}: {e}")

            all_records_map[key] = rec
            # Incremental save
            pd.DataFrame(list(all_records_map.values())).to_csv(OUT_CSV, index=False, encoding="utf-8")

            time.sleep(3.5)

    # Save final
    final_df = pd.DataFrame(list(all_records_map.values()))
    final_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nAll {len(final_df)} records saved to {OUT_CSV}. Summary per file:\n")

    for file_rel, _, _ in FILES_CONFIG:
        df_f = final_df[final_df["file"] == file_rel]
        n_rows = len(df_f)
        n_v_shown = df_f["verdict"].notna().sum()
        n_t_shown = (df_f["type_shown"] == True).sum()
        n_exc = df_f["exception"].notna().sum()
        n_err = df_f["type_error"].notna().sum()

        print(f"=======================================================")
        print(f"File: {file_rel}")
        print(f"Total Rows: {n_rows}")
        print(f"Verdict Shown: {n_v_shown}/{n_rows}")
        print(f"Type Shown: {n_t_shown}/{n_rows}")
        print(f"Exceptions: {n_exc}")
        print(f"Rows with Error Field: {n_err}")
        
        # (verdict, type) counts
        pairs = []
        for _, r in df_f.iterrows():
            v = str(r.get("verdict"))
            t = str(r.get("type_label"))
            pairs.append((v, t))
        counter = Counter(pairs)
        print("Verdict-Type Pairs:")
        for (v, t), cnt in sorted(counter.items()):
            print(f"  ({v}, {t}): {cnt}")

        # Check failing rows
        failing = df_f[(df_f["valid_verdict"] == False) | (df_f["valid_type"] == False) | df_f["exception"].notna()]
        if len(failing) > 0:
            print(f"Failing rows ({len(failing)} total, showing up to 10):")
            for _, r in failing.head(10).iterrows():
                print(f"  ID: {r.get('row_id')}, Text: {str(r.get('text'))[:60]}, ValidV: {r.get('valid_verdict')}, ValidT: {r.get('valid_type')}, Exc: {r.get('exception')}")
        else:
            print("Failing rows: None (0)")
        print()

if __name__ == "__main__":
    main()
