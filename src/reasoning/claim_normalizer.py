import os
import re
import json
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

GROQ_CHAT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


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
    return None


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

    if not raw or not str(raw).strip():
        return default_res

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return default_res

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are a claim normalization assistant. Your task is to process input text and return a JSON object.\n"
        "Instructions:\n"
        "1. Remove OCR junk, site names, watermarks, and boilerplate.\n"
        "2. Fix obvious typos and OCR errors.\n"
        "3. Translate to English if needed.\n"
        "4. Output exactly ONE declarative claim.\n"
        "5. Never add facts or change meaning.\n"
        "6. If the input is already a clean English claim, return it unchanged.\n"
        "Respond with a JSON object in this format:\n"
        '{"claim": "<normalized English claim>", "language": "<detected input language>", "translated": <true|false>}'
    )

    user_prompt = f"Raw text: {raw}"

    # Primary attempt: response_format json_object with sufficient token budget
    payload = {
        "model": GROQ_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_completion_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    parsed = None
    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=12, verify=False)
        if response.status_code == 200:
            raw_resp = response.json()["choices"][0]["message"]["content"].strip()
            parsed = _lenient_json_parse(raw_resp)
        else:
            print(f"[Normalizer 400/Error] status={response.status_code} body={response.text}")
    except Exception as e:
        print(f"[Normalizer Primary Error] {e}")

    # Retry once with minimal prompt and lenient parse if primary failed or parse failed
    if not parsed or not parsed.get("claim"):
        try:
            minimal_payload = {
                "model": GROQ_CHAT_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Clean this text into a single factual claim as JSON "
                            f'{{"claim": "<cleaned English claim>", "language": "English", "translated": false}}:\n{raw}'
                        ),
                    }
                ],
                "temperature": 0.1,
                "max_completion_tokens": 1024,
            }
            retry_resp = requests.post(CHAT_URL, headers=headers, json=minimal_payload, timeout=12, verify=False)
            if retry_resp.status_code == 200:
                raw_resp2 = retry_resp.json()["choices"][0]["message"]["content"].strip()
                parsed = _lenient_json_parse(raw_resp2)
        except Exception:
            pass

    if parsed and parsed.get("claim"):
        norm_claim = str(parsed["claim"]).strip()
        if norm_claim:
            lang = str(parsed.get("language", "English")).strip()
            trans = bool(parsed.get("translated", False))
            return {
                "claim": norm_claim,
                "language": lang,
                "translated": trans,
                "source": "groq",
            }

    # (c) if the model returns an empty claim, keep raw
    return default_res
