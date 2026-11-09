import os
import re
import json
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

GROQ_CHAT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

VALID_LABELS = {"ordinary", "satire", "misleading", "clickbait"}


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


SYSTEM_PROMPT = """You are an expert news analyst specializing in classifying headline and article style.
Classify the given text into exactly ONE of the following four categories:

1. ordinary: Straightforward, factual, conventional news or descriptive headline reporting real events.
   - Example 1: "Central bank holds benchmark interest rate steady at 4.5 percent."
   - Example 2: "Firefighters contain residential blaze in downtown Chicago with no injuries reported."
   - Example 3: "Astronomers detect rare alignment of Jupiter and Saturn visible across North America."

2. satire: Parody, irony, comedic exaggeration, or humor not intended to deceive seriously.
   - Example 1: "Local dog receives honorary master's degree in sleeping through Zoom meetings."
   - Example 2: "Congress passes emergency bill mandating weekend be extended to five days."
   - Example 3: "Area man shocked to learn turning computer off and on again fixed entire universe."

3. misleading: Real photos/events presented with fabricated contexts, altered details, or manipulated assertions.
   - Example 1: "1942 wartime training photograph falsely claimed as live footage of today's conflict."
   - Example 2: "Government announces secret ban on gasoline vehicles starting tomorrow morning."
   - Example 3: "Old festival crowd photo recirculated as protest against new tax law."

4. clickbait: Exaggerated, emotionally charged teasers or hyperbolic phrasing designed solely to trigger clicks.
   - Example 1: "You won't believe what this 90s television star looks like today!"
   - Example 2: "This simple pantry ingredient melts belly fat overnight: doctors are furious."
   - Example 3: "She opened her hotel room door and what happened next shocked the entire internet."

Instructions:
- The input text can be in any language (English, Hindi, Marathi, etc.).
- Always output the label in English as one of: ordinary, satire, misleading, clickbait.
- Provide a brief 1-2 sentence rationale explaining your reasoning.
- Return ONLY a valid JSON object in this format:
{"label": "<ordinary|satire|misleading|clickbait>", "rationale": "<brief explanation>"}
"""


def classify_fake_type(text):
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

    payload = {
        "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text to classify:\n{text}"},
        ],
        "temperature": 0.1,
        "max_completion_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    parsed = None
    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=10, verify=False)
        if response.status_code == 200:
            raw_content = response.json()["choices"][0]["message"]["content"].strip()
            parsed = _lenient_json_parse(raw_content)
    except Exception:
        pass

    # Retry once with minimal prompt on failure or parse error
    if not parsed or parsed.get("label") not in VALID_LABELS:
        try:
            minimal_payload = {
                "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Classify this text as ordinary, satire, misleading, or clickbait. "
                            f'Output JSON {{"label": "<category>", "rationale": "<reason>"}}:\n{text}'
                        ),
                    }
                ],
                "temperature": 0.1,
                "max_completion_tokens": 1024,
            }
            retry_resp = requests.post(CHAT_URL, headers=headers, json=minimal_payload, timeout=10, verify=False)
            if retry_resp.status_code == 200:
                raw_content2 = retry_resp.json()["choices"][0]["message"]["content"].strip()
                parsed = _lenient_json_parse(raw_content2)
        except Exception:
            pass

    if parsed and isinstance(parsed, dict):
        raw_label = str(parsed.get("label", "")).strip().lower()
        if raw_label in VALID_LABELS:
            rationale = str(parsed.get("rationale", "")).strip() or "Classified by Groq few-shot model."
            return {
                "label": raw_label,
                "rationale": rationale
            }

    return default_res
