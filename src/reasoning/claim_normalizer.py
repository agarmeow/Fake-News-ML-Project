import os
import json
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def normalize_claim(raw):
    """
    normalize_claim(raw) -> {"claim", "language", "translated", "source"}
    source = "groq" or "raw"
    """
    default_res = {
        "claim": raw,
        "language": "unknown",
        "translated": False,
        "source": "raw"
    }

    if not raw or not raw.strip() or not GROQ_API_KEY:
        return default_res

    system_prompt = (
        "You are a claim normalization assistant. Your task is to process input text and return a JSON object.\n"
        "Instructions:\n"
        "1. Remove OCR junk, site names, watermarks, and boilerplate.\n"
        "2. Fix obvious typos and OCR errors (e.g., 'capital city in Mumbai' -> 'capital city is Mumbai').\n"
        "3. Translate to English if needed.\n"
        "4. Output exactly ONE declarative claim.\n"
        "5. Never add facts or change meaning.\n"
        "6. If the input is already a clean English claim, return it unchanged.\n"
        "Respond ONLY with a valid JSON object in this exact schema:\n"
        '{"claim": "<normalized English claim>", "language": "<detected input language>", "translated": <true|false>}'
    )

    user_prompt = f"Raw text: {raw}"

    payload = {
        "model": GROQ_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_completion_tokens": 200,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=10, verify=False)
        response.raise_for_status()

        raw_resp = response.json()["choices"][0]["message"]["content"].strip()
        if raw_resp.startswith("```"):
            raw_resp = raw_resp.strip("`")
            if raw_resp.lower().startswith("json"):
                raw_resp = raw_resp.split("\n", 1)[-1]

        data = json.loads(raw_resp)
        normalized_claim = str(data.get("claim", "")).strip()
        if not normalized_claim:
            return default_res

        language = str(data.get("language", "unknown")).strip()
        translated = bool(data.get("translated", False))

        return {
            "claim": normalized_claim,
            "language": language,
            "translated": translated,
            "source": "groq"
        }
    except Exception:
        return default_res
