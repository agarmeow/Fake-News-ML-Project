import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from src.reasoning.faketype_pipeline import classify_type
except ImportError:
    from reasoning.faketype_pipeline import classify_type


def is_english(lang, text):
    l_clean = str(lang).strip().lower() if lang is not None else ""
    if l_clean == "en" or l_clean.startswith("en") or l_clean == "english":
        return True
    if l_clean in ("", "unknown", "none"):
        text_str = str(text) if text is not None else ""
        alpha_chars = [c for c in text_str if c.isalpha()]
        if not alpha_chars:
            return False
        ascii_alpha = [c for c in alpha_chars if "a" <= c.lower() <= "z"]
        return (len(ascii_alpha) / len(alpha_chars)) >= 0.90
    return False


def get_type_view(claim_text, verifier_result, language) -> dict:
    """
    get_type_view(claim_text, verifier_result, language) -> dict
    keys: shown (bool), label, source, trained_label, trained_conf,
          llm_label, llm_conf, uncertain, error.
    """
    try:
        if not is_english(language, claim_text):
            return {
                "shown": False,
                "label": None,
                "source": None,
                "trained_label": None,
                "trained_conf": None,
                "llm_label": None,
                "llm_conf": None,
                "uncertain": None,
                "error": None,
            }

        # For English call classify_type with a verifier_fn that returns the ALREADY computed verifier_result
        res = classify_type(claim_text, verifier_fn=lambda _: verifier_result)

        return {
            "shown": True,
            "label": res.get("label"),
            "source": res.get("source"),
            "trained_label": res.get("trained_label"),
            "trained_conf": res.get("trained_conf"),
            "llm_label": res.get("llm_label"),
            "llm_conf": res.get("llm_conf"),
            "uncertain": res.get("uncertain"),
            "error": None,
        }
    except Exception as e:
        return {
            "shown": False,
            "label": None,
            "source": None,
            "trained_label": None,
            "trained_conf": None,
            "llm_label": None,
            "llm_conf": None,
            "uncertain": None,
            "error": str(e),
        }
