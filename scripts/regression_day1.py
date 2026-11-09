import sys
import os
import time

# Reconfigure stdout/stderr to UTF-8 to prevent Windows charmap encoding crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# Ensure local offline cache is prioritized for Hugging Face Transformers
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fact_check.fact_checker import fact_check
from verification.consistency_checker import run_tests as run_consistency_tests

TEST_CLAIMS = [
    {
        "claim": "India's capital is New Delhi",
        "expected_check": lambda r: r and r.get("verdict") == "TRUE",
        "desc": "-> TRUE"
    },
    {
        "claim": "India's capital is Mumbai",
        "expected_check": lambda r: r and r.get("verdict") == "FALSE",
        "desc": "-> FALSE"
    },
    {
        "claim": "India's capital is Delhi",
        "expected_check": lambda r: r and r.get("verdict") == "TRUE",
        "desc": "-> TRUE"
    },
    {
        "claim": "India's capital city in Mumbai",
        "expected_check": lambda r: r and (r.get("verdict") != "TRUE" or r.get("conflict") is True),
        "desc": "-> must not be TRUE (FALSE, or conflict flag set)"
    },
    {
        "claim": "Music icon Dolly Parton has died at 80",
        "eval_fn": lambda r: (
            "PASS" if r and r.get("verdict") == "TRUE" and r.get("verdict_source") in {"rule_based", "llm_fallback"}
            else "FAIL"
        ),
        "desc": "-> TRUE (rule_based or llm_fallback)"
    },
    {
        "claim": "Jamie Chastain to tour Liverpool with Borcelle in 2023",
        "eval_fn": lambda r: (
            "PASS" if r and r.get("verdict") == "UNVERIFIABLE"
            else "FAIL"
        ),
        "desc": "-> UNVERIFIABLE (FALSE or TRUE = FAIL)"
    },
]


def run_regression():
    all_passed = True
    results = []

    print(f"{'CLAIM':<55} | {'VERDICT':<13} | {'CONF':<6} | {'SOURCE':<12} | {'SEC':<6} | {'STATUS'}")
    print("-" * 105)

    for item in TEST_CLAIMS:
        claim = item["claim"]
        t0 = time.time()
        try:
            res = fact_check(claim)
        except Exception as e:
            res = {"verdict": f"ERR: {e}", "confidence": 0.0, "verdict_source": "error"}
        dt = time.time() - t0

        verdict = res.get("verdict", "N/A") if res else "None"
        conf = res.get("confidence", 0.0) if res else 0.0
        v_src = res.get("verdict_source", "N/A") if res else "N/A"

        if "eval_fn" in item:
            status = item["eval_fn"](res)
            if status == "FAIL":
                all_passed = False
        else:
            passed = item["expected_check"](res)
            if not passed:
                all_passed = False
            status = "PASS" if passed else "FAIL"

        print(f"{claim:<55} | {verdict:<13} | {conf:<6.2f} | {v_src:<12} | {dt:<6.2f} | {status}")
        results.append({
            "claim": claim,
            "verdict": verdict,
            "confidence": conf,
            "verdict_source": v_src,
            "seconds": dt,
            "status": status
        })

    print("-" * 105)
    print("\nRunning Consistency Test Suite...")
    try:
        run_consistency_tests()
    except Exception as e:
        print(f"Consistency suite error: {e}")
        all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_regression()
    if not success:
        sys.exit(1)
    sys.exit(0)
