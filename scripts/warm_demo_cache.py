import sys
import os
import time
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

has_groq = bool(os.getenv("GROQ_API_KEY"))
print(f"GROQ_API_KEY set: {has_groq}")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fact_check.fact_checker import fact_check


def main():
    claims_file = os.path.join(PROJECT_ROOT, "data", "demo_claims.txt")
    if not os.path.exists(claims_file):
        print(f"Error: {claims_file} not found.")
        sys.exit(1)

    with open(claims_file, "r", encoding="utf-8") as f:
        claims = [line.strip() for line in f if line.strip()]

    print(f"\nWarming demo cache for {len(claims)} claims...\n")
    header = f"{'CLAIM':<55} | {'EVID':<5} | {'SOURCE':<12} | {'VERDICT':<13} | {'V_SRC':<12} | {'SEC':<6}"
    print(header)
    print("-" * len(header))

    any_zero_evidence = False

    for claim in claims:
        t0 = time.time()
        try:
            res = fact_check(claim)
        except Exception as e:
            res = {
                "evidence": [],
                "retrieval_source": "error",
                "verdict": f"ERR: {e}",
                "verdict_source": "error"
            }
        dt = time.time() - t0

        ev_count = len(res.get("evidence", [])) if res else 0
        ret_src = res.get("retrieval_source", "none") if res else "none"
        verdict = res.get("verdict", "none") if res else "none"
        v_src = res.get("verdict_source", "none") if res else "none"

        if ev_count == 0:
            any_zero_evidence = True

        print(f"{claim:<55} | {ev_count:<5} | {ret_src:<12} | {verdict:<13} | {v_src:<12} | {dt:<6.2f}")

    print("-" * len(header))

    if any_zero_evidence:
        print("\nFAILED: One or more demo claims have 0 evidence.")
        sys.exit(1)
    else:
        print("\nSUCCESS: All demo claims have cached evidence.")
        sys.exit(0)


if __name__ == "__main__":
    main()
