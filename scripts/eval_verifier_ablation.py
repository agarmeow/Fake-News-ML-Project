import os
import sys
import time
import math
import json
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout/stderr for Unicode safety on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY set: {bool(groq_key)}")
if not groq_key:
    print("FATAL: GROQ_API_KEY is not set! Aborting Groq steps loudly.")
    sys.exit(1)

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reasoning.claim_normalizer import normalize_claim
from reasoning.llm_reasoner import llm_verify
from fact_check.fact_checker import (
    fact_check,
    evaluate_evidence,
    aggregate_results,
    build_alternative_query,
    retrieve_evidence,
    get_cached_models,
    QUALITY_MAP,
    RELEVANCE_THRESHOLD,
)

DEBUG_IDS = {
    "T01", "T02", "T03", "T04", "T05", "T06",
    "F01", "F02", "F03", "F04", "F05", "F06",
}

CONSECUTIVE_GROQ_FAILURES = 0


def clean_verdict(v: str) -> str:
    if not isinstance(v, str):
        return "UNVERIFIABLE"
    vu = v.strip().upper()
    if vu in ("TRUE", "FALSE", "UNVERIFIABLE"):
        return vu
    if "TRUE" in vu:
        return "TRUE"
    if "FALSE" in vu:
        return "FALSE"
    return "UNVERIFIABLE"


def _lenient_json_parse(text: str):
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s.split("\n", 1)[-1]
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*?\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def call_c0_closed_book(claim_n: str) -> dict:
    """
    C0 closed-book LLM: Groq (GROQ_FAKETYPE_MODEL, temperature 0, JSON) with NO evidence;
    prompt: "Using only your own knowledge, label the claim TRUE, FALSE or UNVERIFIABLE.
    If you are not sure, or it may concern events after your training data, answer UNVERIFIABLE."
    Return {"verdict","confidence"}.
    """
    global CONSECUTIVE_GROQ_FAILURES

    if CONSECUTIVE_GROQ_FAILURES >= 5:
        print("FATAL: 5 consecutive Groq failures reached. Aborting loudly!")
        sys.exit(1)

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_FAKETYPE_MODEL", "openai/gpt-oss-120b").strip()

    prompt = (
        "Using only your own knowledge, label the claim TRUE, FALSE or UNVERIFIABLE. "
        "If you are not sure, or it may concern events after your training data, answer UNVERIFIABLE. "
        "Respond strictly in valid JSON with keys 'verdict' and 'confidence' (float 0.0-1.0)."
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Claim: {claim_n}"},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 100,
    }

    models_to_try = [model, "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
    for curr_model in models_to_try:
        payload["model"] = curr_model
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=15, verify=False)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after")
                    err_lower = resp.text.lower()
                    if "daily" in err_lower or (retry_after and int(retry_after) > 60):
                        print(f"Warning: Rate limit / daily limit on {curr_model}. Trying fallback model.")
                        break  # try next fallback model
                    time.sleep(min(2 ** (attempt + 1), 20))
                    continue

                if resp.status_code in (403, 404):
                    break

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = _lenient_json_parse(content)

                if parsed and "verdict" in parsed:
                    v = clean_verdict(str(parsed["verdict"]))
                    try:
                        conf = float(parsed.get("confidence", 0.8))
                    except Exception:
                        conf = 0.8
                    CONSECUTIVE_GROQ_FAILURES = 0
                    return {"verdict": v, "confidence": conf}

            except Exception as e:
                print(f"Warning in C0 call (model={curr_model}, attempt={attempt}): {e}")
                time.sleep(1)

    CONSECUTIVE_GROQ_FAILURES += 1
    print(f"Warning: C0 failed across fallback models. Consecutive failures: {CONSECUTIVE_GROQ_FAILURES}")
    return {"verdict": "UNVERIFIABLE", "confidence": 0.0}


def eval_c1_nli_only(results: list) -> tuple:
    """
    C1 NLI only: primary evidence; SUPPORT when NLI label is entailment,
    CONTRADICT when contradiction, same weights (similarity x source quality x NLI confidence)
    and the same relevance threshold as aggregate_results; no consistency checker,
    no entity guard, no keyword guard, no competing pass, no LLM.
    Verdict TRUE/FALSE/UNVERIFIABLE by the same decision rule.
    """
    support_score = 0.0
    contradiction_score = 0.0

    for result in results:
        semantic_similarity = max(result.get("semantic_similarity", 0.0), 0.0)
        if semantic_similarity < RELEVANCE_THRESHOLD:
            continue

        sq_val = result.get("source_quality", 0.5)
        if isinstance(sq_val, str):
            source_quality = QUALITY_MAP.get(sq_val.lower(), 0.5)
        else:
            try:
                source_quality = float(sq_val)
            except (TypeError, ValueError):
                source_quality = 0.5
        source_quality = max(source_quality, 0.0)

        nli_confidence = max(result.get("nli_confidence", 0.0), 0.0)
        nli_label = str(result.get("nli_label", "")).lower()

        evidence_weight = semantic_similarity * source_quality * nli_confidence

        if "entail" in nli_label:
            support_score += evidence_weight
        elif "contradict" in nli_label:
            contradiction_score += evidence_weight

    total_decisive_score = support_score + contradiction_score

    if total_decisive_score == 0:
        return "UNVERIFIABLE", 0.0
    elif support_score > contradiction_score:
        dominant = support_score
        agreement = dominant / total_decisive_score
        strength = 1.0 - math.exp(-total_decisive_score)
        conf = min(0.99, agreement * strength)
        return "TRUE", conf
    elif contradiction_score > support_score:
        dominant = contradiction_score
        agreement = dominant / total_decisive_score
        strength = 1.0 - math.exp(-total_decisive_score)
        conf = min(0.99, agreement * strength)
        return "FALSE", conf
    else:
        return "UNVERIFIABLE", 0.0


def main():
    parser = argparse.ArgumentParser(description="Verifier Ablation Evaluation")
    parser.add_argument("--debug", action="store_true", help="Run on DEBUG rows only (T01-T06, F01-F06)")
    args = parser.parse_args()

    claims_csv = PROJECT_ROOT / "data" / "eval" / "claims.csv"
    if not claims_csv.exists():
        print(f"FATAL: {claims_csv} does not exist!")
        sys.exit(1)

    out_csv = PROJECT_ROOT / "results" / "verifier_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(claims_csv)
    if args.debug:
        eval_df = df[df["id"].isin(DEBUG_IDS)].copy().reset_index(drop=True)
        print(f"Running DEBUG mode on {len(eval_df)} rows: {eval_df['id'].tolist()}")
    else:
        eval_df = df.copy().reset_index(drop=True)
        print(f"Running FULL mode on {len(eval_df)} rows.")

    # Resume from existing results if file exists
    records = []
    done_ids = set()
    if out_csv.exists():
        try:
            prev_df = pd.read_csv(out_csv)
            if len(prev_df) > 0 and "id" in prev_df.columns:
                records = prev_df.to_dict("records")
                done_ids = set(r["id"] for r in records)
                print(f"Resuming: found {len(done_ids)} finished ids in {out_csv.name}.")
        except Exception as e:
            print(f"Warning reading existing CSV: {e}")
            records = []

    # Preload models once
    print("Loading models (reused across all rows)...")
    semantic_model, tokenizer, nli_model = get_cached_models()

    n_total = len(eval_df)
    for idx, row in eval_df.iterrows():
        row_id = str(row["id"]).strip()
        if row_id in done_ids:
            continue

        raw_claim = str(row["claim"]).strip()
        gold = clean_verdict(str(row["label"]))
        category = str(row.get("category", "")).strip()
        language = str(row.get("language", "en")).strip()

        print(f"\n[{idx+1}/{n_total}] Processing ID: {row_id} ({category}, {language}) | Gold: {gold}")
        print(f"Raw claim: {raw_claim[:70]}...")

        # 1. Normalize claim
        norm_res = normalize_claim(raw_claim)
        claim_n = norm_res.get("claim", raw_claim)
        norm_source = norm_res.get("source", "rule_based")
        print(f"Normalized: {claim_n[:70]}... (source: {norm_source})")

        # 2. C0: Closed-book LLM (no evidence)
        c0_res = call_c0_closed_book(claim_n)
        c0_verdict = c0_res["verdict"]
        c0_conf = c0_res["confidence"]
        print(f"  C0 (Closed-book): {c0_verdict} ({c0_conf:.2f})")

        # 3. Retrieve primary evidence once
        evidence = retrieve_evidence(claim_n, max_results=5)
        primary_results = evaluate_evidence(
            claim_n, evidence, tokenizer, nli_model, semantic_model
        ) if evidence else []

        # 4. C1: NLI only
        c1_verdict, c1_conf = eval_c1_nli_only(primary_results)
        print(f"  C1 (NLI only): {c1_verdict} ({c1_conf:.2f})")

        # 5. C2: C1 + consistency checker (primary evidence only)
        if primary_results:
            c2_agg = aggregate_results(primary_results)
            c2_verdict = clean_verdict(c2_agg[0])
            c2_conf = float(c2_agg[1])
        else:
            c2_verdict, c2_conf = "UNVERIFIABLE", 0.0
        print(f"  C2 (NLI + Consistency): {c2_verdict} ({c2_conf:.2f})")

        # 6. C3: C2 + competing-fact retrieval pass (only when primary rule verdict is UNVERIFIABLE)
        c3_verdict, c3_conf = c2_verdict, c2_conf
        if c2_verdict == "UNVERIFIABLE" and primary_results:
            claim_relation = primary_results[0].get("claim_relation")
            alt_query = build_alternative_query(claim_relation)
            if alt_query:
                alt_evidence = retrieve_evidence(alt_query, max_results=5)
                seen_urls = {r["url"] for r in primary_results}
                alt_evidence = [e for e in alt_evidence if e.get("url") not in seen_urls]
                if alt_evidence:
                    alt_results = evaluate_evidence(
                        claim_n, alt_evidence, tokenizer, nli_model, semantic_model
                    )
                    merged = primary_results + alt_results
                    c3_agg = aggregate_results(merged)
                    c3_verdict = clean_verdict(c3_agg[0])
                    c3_conf = float(c3_agg[1])
        print(f"  C3 (C2 + Competing-fact): {c3_verdict} ({c3_conf:.2f})")

        # 7. C4: Full system as shipped
        t_c4_start = time.time()
        c4_fc = fact_check(claim_n)
        c4_secs = time.time() - t_c4_start
        if c4_fc:
            c4_verdict = clean_verdict(c4_fc.get("verdict"))
            c4_conf = float(c4_fc.get("confidence", 0.0))
            c4_source = c4_fc.get("verdict_source", "rule_based")
            c4_conflict = bool(c4_fc.get("conflict", False))
        else:
            c4_verdict, c4_conf, c4_source, c4_conflict = "UNVERIFIABLE", 0.0, "rule_based", False
        print(f"  C4 (Full system): {c4_verdict} ({c4_conf:.2f}, src={c4_source}, conflict={c4_conflict}, {c4_secs:.2f}s)")

        # 8. C5: LLM reading primary evidence, no rules
        try:
            c5_llm = llm_verify(claim_n, primary_results) if primary_results else None
            if c5_llm:
                c5_verdict = clean_verdict(c5_llm.get("verdict"))
                c5_conf = float(c5_llm.get("confidence", 0.0))
            else:
                c5_verdict, c5_conf = "UNVERIFIABLE", 0.0
        except Exception as e:
            print(f"Warning in C5 llm_verify: {e}")
            c5_verdict, c5_conf = "UNVERIFIABLE", 0.0
        print(f"  C5 (LLM + primary evidence): {c5_verdict} ({c5_conf:.2f})")

        rec = {
            "id": row_id,
            "category": category,
            "language": language,
            "gold": gold,
            "claim_n": claim_n,
            "normalizer_source": norm_source,
            "c0_verdict": c0_verdict,
            "c0_confidence": c0_conf,
            "c1_verdict": c1_verdict,
            "c1_confidence": c1_conf,
            "c2_verdict": c2_verdict,
            "c2_confidence": c2_conf,
            "c3_verdict": c3_verdict,
            "c3_confidence": c3_conf,
            "c4_verdict": c4_verdict,
            "c4_confidence": c4_conf,
            "c5_verdict": c5_verdict,
            "c5_confidence": c5_conf,
            "c4_verdict_source": c4_source,
            "c4_conflict": c4_conflict,
            "c4_seconds": round(c4_secs, 2),
        }
        records.append(rec)
        done_ids.add(row_id)
        pd.DataFrame(records).to_csv(out_csv, index=False)

    print(f"\nExecution finished. Total records written: {len(records)}")


if __name__ == "__main__":
    main()
