import os
import sys
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from fact_check.fact_checker import (
    retrieve_evidence,
    evaluate_evidence,
    aggregate_results,
    get_cached_models,
    get_primary_entity,
    check_entity_in_passage,
    RELATION_MAP,
)

claims_csv = PROJECT_ROOT / "data" / "eval" / "claims.csv"
df = pd.read_csv(claims_csv)

sem, tok, nli = get_cached_models()

for target_id in ["F07", "F11"]:
    row = df[df["id"] == target_id].iloc[0]
    claim = str(row["claim"]).strip()
    gold = str(row["label"]).strip()

    print("=" * 80)
    print(f"DIAGNOSTIC V6b: ID {target_id} | Gold: {gold}")
    print(f"Claim: {claim}")

    primary_ent = get_primary_entity(claim)
    entity_guard_ran = primary_ent is not None
    print(f"Primary entity extracted: {primary_ent} ({primary_ent.label_ if primary_ent else 'None'})")
    print(f"Entity guard status: {'RAN' if entity_guard_ran else 'SKIPPED (no named entity)'}")

    ev = retrieve_evidence(claim, max_results=5)
    results = evaluate_evidence(claim, ev, tok, nli, sem) if ev else []

    # Run aggregate_results to populate counted_as and scores
    agg_res = aggregate_results(results)
    verdict, confidence, sup_score, contra_score = agg_res[0], agg_res[1], agg_res[2], agg_res[3]

    print(f"\nVerdict: {verdict} (Confidence: {confidence:.4f}) | Support score: {sup_score:.4f} | Contradiction score: {contra_score:.4f}")
    print(f"Rule path: {'support_score > contradiction_score -> TRUE' if sup_score > contra_score else 'other'}")

    print("\nPER-EVIDENCE TABLE:")
    print(f"{'Title':<30} | {'NLI':<10} | {'Conf':<6} | {'Consist':<10} | {'EntCons':<7} | {'ClaimRel':<10} | {'EvRel':<10} | {'Abst':<5} | {'EntOK':<5} | {'KwGuard':<7} | {'CountedAs':<9}")
    print("-" * 125)

    for r in results:
        t = str(r.get("title", ""))[:28]
        nli_l = str(r.get("nli_label", ""))
        nli_c = float(r.get("nli_confidence", 0.0))
        c_v = str(r.get("consistency_verdict", ""))
        e_c = float(r.get("entity_consistency", 0.0))
        c_rel = str(r.get("claim_relation"))[:10]
        e_rel = str(r.get("evidence_relation"))[:10]

        # Check abstention & keyword guard
        abstained = r.get("claim_relation") is None
        kw_guard = False
        if abstained:
            c_str = r.get("claim", "")
            if c_str:
                claim_tokens = {w.lower().strip(".,'\"?!;:()[]{}") for w in c_str.split()}
                trigger_words = set(RELATION_MAP.keys())
                if claim_tokens.intersection(trigger_words):
                    kw_guard = True
                    abstained = False

        passage = r.get("evidence", "")
        ent_ok = check_entity_in_passage(primary_ent, passage)
        cnt_as = str(r.get("counted_as", "none"))

        print(f"{t:<30} | {nli_l:<10} | {nli_c:<6.2f} | {c_v:<10} | {e_c:<7.2f} | {c_rel:<10} | {e_rel:<10} | {str(abstained):<5} | {str(ent_ok):<5} | {str(kw_guard):<7} | {cnt_as:<9}")
