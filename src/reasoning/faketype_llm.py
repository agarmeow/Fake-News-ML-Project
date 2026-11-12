import os
import re
import sys
import time
import json
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

GROQ_CHAT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

VALID_LABELS = {"ordinary", "satire", "misleading", "clickbait"}

LABEL_SYNONYMS = {
    "ordinary news": "ordinary",
    "ordinary": "ordinary",
    "real": "ordinary",
    "factual": "ordinary",
    "conventional": "ordinary",
    "humor": "satire",
    "parody": "satire",
    "satire": "satire",
    "irony": "satire",
    "click bait": "clickbait",
    "click-bait": "clickbait",
    "clickbait": "clickbait",
    "hyperbolic": "clickbait",
    "sensational": "clickbait",
    "teaser": "clickbait",
    "fake": "misleading",
    "fabricated": "misleading",
    "manipulated": "misleading",
    "misleading news": "misleading",
    "misleading": "misleading",
}


def normalize_label(raw):
    if not raw:
        return None
    cleaned = str(raw).strip().lower()
    if cleaned in VALID_LABELS:
        return cleaned
    if cleaned in LABEL_SYNONYMS:
        return LABEL_SYNONYMS[cleaned]
    for k, v in LABEL_SYNONYMS.items():
        if k in cleaned:
            return v
    for v in VALID_LABELS:
        if v in cleaned:
            return v
    return None


def _lenient_json_parse(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text.split("\n", 1)[-1]
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Regex extraction fallback
    lbl_m = re.search(r'"label"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    rat_m = re.search(r'"rationale"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if lbl_m:
        return {
            "label": lbl_m.group(1),
            "rationale": rat_m.group(1) if rat_m else "Classified by Groq."
        }
    return None


SYSTEM_PROMPT = """You are an expert news analyst. Classify the headline or text into exactly ONE category:
1. ordinary: Straightforward, factual news or descriptive headline reporting real events.
2. satire: Parody, irony, comedic exaggeration, or humor not intended to deceive seriously.
3. misleading: Fabricated assertions, manipulated contexts, fake quotes, or deceptive claims.
4. clickbait: Exaggerated teasers or sensationalized emotional hooks designed to bait clicks.

Rules:
- Works for any language (English, Hindi, Marathi, etc.).
- Output ONLY valid JSON:
{"label": "<ordinary|satire|misleading|clickbait>", "rationale": "<1-2 sentence explanation>"}
"""


def classify_fake_type(text, max_retries=5):
    """
    classify_fake_type(text) -> {"label", "rationale"}
    labels: ordinary | satire | misleading | clickbait | unclear
    """
    default_res = {
        "label": "unclear",
        "rationale": "Classification unavailable or inconclusive."
    }

    if not text or not str(text).strip():
        return default_res

    # ENV RULE: read GROQ_API_KEY at call time, not at import
    api_key = os.environ.get("GROQ_API_KEY")
    key_is_set = bool(api_key and api_key.strip())
    print(f"GROQ_API_KEY set: {key_is_set}")

    if not key_is_set:
        raise RuntimeError("GROQ_API_KEY is not set in environment! Aborting Groq steps.")

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text to classify:\n{text}"},
        ],
        "temperature": 0.0,
        "max_completion_tokens": 200,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=15, verify=False)
            if response.status_code == 200:
                raw_content = response.json()["choices"][0]["message"]["content"].strip()
                parsed = _lenient_json_parse(raw_content)
                if parsed and isinstance(parsed, dict):
                    norm_lbl = normalize_label(parsed.get("label"))
                    if norm_lbl:
                        rat = str(parsed.get("rationale", "")).strip() or "Classified by Groq few-shot model."
                        return {"label": norm_lbl, "rationale": rat}
            elif response.status_code == 429:
                # Rate limited - exponential backoff with jitter
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    wait_s = float(retry_after) + 0.5
                else:
                    wait_s = min(30, (2 ** attempt) + 1.5)
                time.sleep(wait_s)
                continue
            else:
                time.sleep(1.0)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(min(15, (2 ** attempt) + 1.0))
            continue
        except Exception:
            time.sleep(1.0)

    # Secondary fallback with concise prompt if first loop exhausted
    try:
        fallback_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Classify as ordinary, satire, misleading, or clickbait.\n"
                        f'{{"label": "<category>", "rationale": "<reason>"}}\nText: {text}'
                    ),
                }
            ],
            "temperature": 0.0,
            "max_completion_tokens": 150,
        }
        resp = requests.post(CHAT_URL, headers=headers, json=fallback_payload, timeout=12, verify=False)
        if resp.status_code == 200:
            raw2 = resp.json()["choices"][0]["message"]["content"].strip()
            parsed2 = _lenient_json_parse(raw2)
            if parsed2 and isinstance(parsed2, dict):
                norm_lbl = normalize_label(parsed2.get("label"))
                if norm_lbl:
                    rat = str(parsed2.get("rationale", "")).strip() or "Classified by Groq."
                    return {"label": norm_lbl, "rationale": rat}
    except Exception:
        pass

    return default_res
