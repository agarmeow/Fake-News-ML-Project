import os
import sys
import re
import json
import time
import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# Track consecutive failures globally to abort after 5 consecutive failures
CONSECUTIVE_FAILURES = 0

FALLBACK_MODELS = [
    os.getenv("GROQ_FAKETYPE_MODEL", "openai/gpt-oss-120b"),
    os.getenv("GROQ_FAKETYPE_MODEL_2", "qwen/qwen3.8-27b"),
    "openai/gpt-oss-20b",
]

FEW_SHOT_SYSTEM_PROMPT = """You are an expert news and headline classifier. Classify the given headline into exactly one of four categories:
1. satire: deadpan, absurd or humorous fabricated news not meant to be believed.
2. clickbait: sensational headline built to provoke a click (curiosity gap, listicle, "you won't believe", withheld detail).
3. misleading: serious, factual-sounding claim that is likely false, exaggerated or unsupported (hoax, viral rumour, miracle cure, fake official announcement).
4. ordinary: plain factual news.

Examples:
- "Federal Reserve maintains benchmark interest rate following two-day policy meeting" -> {"label": "ordinary", "confidence": 0.95, "cues": ["federal reserve", "interest rate", "factual reporting"]}
- "Municipal water authority announces scheduled maintenance for northern district pipelines" -> {"label": "ordinary", "confidence": 0.95, "cues": ["scheduled maintenance", "municipal announcement"]}
- "Department of Transportation completes highway resurfacing project ahead of schedule" -> {"label": "ordinary", "confidence": 0.95, "cues": ["highway project", "department of transportation"]}
- "Retail sales rose 0.4 percent in October led by automotive and grocery purchases" -> {"label": "ordinary", "confidence": 0.95, "cues": ["retail sales", "economic statistics"]}

- "Nation's dogs hold emergency summit to discuss horrifying implications of vacuum cleaner upgrade" -> {"label": "satire", "confidence": 0.98, "cues": ["dogs emergency summit", "vacuum cleaner", "absurd"]}
- "Local man bravely spends entire workday pretending spreadsheet contains urgent data" -> {"label": "satire", "confidence": 0.95, "cues": ["local man bravely", "pretending spreadsheet", "humorous parody"]}
- "Archaeologists uncover ancient civilization that wisely chose not to invent emails" -> {"label": "satire", "confidence": 0.96, "cues": ["ancient civilization", "chose not to invent emails", "irony"]}
- "Area cat demands entry through closed door only to stare blankly at newly revealed room" -> {"label": "satire", "confidence": 0.95, "cues": ["area cat", "stare blankly", "classic parody trope"]}

- "She opened her attic floorboards after 20 years and what was hidden inside will leave you speechless" -> {"label": "clickbait", "confidence": 0.95, "cues": ["what was hidden", "leave you speechless", "curiosity gap"]}
- "Doctors are begging people to stop eating this one everyday vegetable immediately" -> {"label": "clickbait", "confidence": 0.95, "cues": ["doctors begging", "this one vegetable", "sensationalized urgency"]}
- "17 mind-blowing kitchen hacks that will completely change the way you cook pasta" -> {"label": "clickbait", "confidence": 0.95, "cues": ["17 hacks", "mind-blowing", "listicle"]}
- "He looked out his airplane window and couldn't believe his eyes when he saw this" -> {"label": "clickbait", "confidence": 0.95, "cues": ["couldn't believe his eyes", "saw this", "withheld detail"]}

- "Secret internal memo confirms government will permanently shut down power grid next Monday" -> {"label": "misleading", "confidence": 0.95, "cues": ["secret memo", "permanently shut down grid", "conspiracy hoax"]}
- "Drinking boiled garlic and lemon water cures all stages of respiratory viral illness in 12 hours" -> {"label": "misleading", "confidence": 0.95, "cues": ["boiled garlic cures", "miracle cure claim", "12 hours"]}
- "NASA scientists confirm mysterious asteroid will definitely strike Earth by next weekend" -> {"label": "misleading", "confidence": 0.95, "cues": ["definitely strike earth", "next weekend", "fabricated catastrophe"]}
- "UN officially issues global emergency order freezing all personal bank accounts worldwide" -> {"label": "misleading", "confidence": 0.95, "cues": ["freezing personal bank accounts", "fake official order"]}

Respond strictly in valid JSON format:
{
  "label": "<ordinary|satire|clickbait|misleading>",
  "confidence": <float between 0.0 and 1.0>,
  "cues": [<list of strings>]
}
"""


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
    # Regex extraction fallback
    lbl_m = re.search(r'"label"\s*:\s*"([^"]+)"', s, re.IGNORECASE)
    conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', s)
    if lbl_m:
        conf = float(conf_m.group(1)) if conf_m else 0.8
        return {"label": lbl_m.group(1), "confidence": conf, "cues": []}
    return None


def classify(text: str, model: str = None) -> dict:
    """
    classify(text, model=None) -> {"label", "confidence", "cues"}
    Labels: ordinary | satire | clickbait | misleading
    """
    global CONSECUTIVE_FAILURES

    if CONSECUTIVE_FAILURES >= 5:
        print("FATAL: 5 consecutive Groq failures. Aborting Groq steps loudly!")
        sys.exit(1)

    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    key_is_set = bool(groq_key and groq_key.strip())
    # ENV RULE: print "GROQ_API_KEY set: True/False"
    print(f"GROQ_API_KEY set: {key_is_set}")

    if not key_is_set:
        print("FATAL: GROQ_API_KEY is not set! Aborting Groq steps loudly.")
        sys.exit(1)

    # Candidate models to try if 404/403/model-not-found occurs
    if model:
        models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    else:
        m1 = os.getenv("GROQ_FAKETYPE_MODEL", "openai/gpt-oss-120b")
        m2 = os.getenv("GROQ_FAKETYPE_MODEL_2", "qwen/qwen3.8-27b")
        models_to_try = [m1, m2] + [m for m in FALLBACK_MODELS if m not in (m1, m2)]

    headers = {
        "Authorization": f"Bearer {groq_key.strip()}",
        "Content-Type": "application/json",
    }
    url = "https://api.groq.com/openai/v1/chat/completions"

    for current_model in models_to_try:
        model_not_found = False
        for attempt in range(3):
            payload = {
                "model": current_model,
                "messages": [
                    {"role": "system", "content": FEW_SHOT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Headline to classify:\n{text}"},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "max_tokens": 200,
            }

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=15, verify=False)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after")
                    err_lower = resp.text.lower()
                    if "daily" in err_lower or (retry_after and int(retry_after) > 60):
                        print(f"FATAL: Rate limit exceeded or daily limit on {current_model}: {resp.text}")
                        CONSECUTIVE_FAILURES += 1
                        sys.exit(1)
                    wait_s = min(2 ** (attempt + 1) * 2, 20)
                    time.sleep(wait_s)
                    continue

                if resp.status_code in (404, 403) or "model_not_found" in resp.text:
                    print(f"Warning: Model {current_model} returned {resp.status_code}. Falling back to next model.")
                    model_not_found = True
                    break  # Fall back to next model in models_to_try

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()

                parsed = _lenient_json_parse(content)
                if not parsed or "label" not in parsed:
                    # One retry with explicit formatting instruction
                    payload["messages"].append({"role": "assistant", "content": content})
                    payload["messages"].append({"role": "user", "content": "Provide only valid JSON with keys label, confidence, cues."})
                    retry_resp = requests.post(url, headers=headers, json=payload, timeout=15, verify=False)
                    if retry_resp.status_code == 200:
                        parsed = _lenient_json_parse(retry_resp.json()["choices"][0]["message"]["content"].strip())

                if parsed and "label" in parsed:
                    lbl = str(parsed.get("label", "")).strip().lower()
                    if lbl in ("ordinary news", "plain", "factual", "normal"):
                        lbl = "ordinary"
                    elif lbl not in ("ordinary", "satire", "clickbait", "misleading"):
                        lbl = "ordinary"

                    conf = float(parsed.get("confidence", 0.8))
                    cues = parsed.get("cues", [])
                    if isinstance(cues, str):
                        cues = [cues]

                    CONSECUTIVE_FAILURES = 0
                    return {"label": lbl, "confidence": conf, "cues": cues, "model": data.get("model", current_model)}

            except requests.exceptions.Timeout:
                wait_s = min(2 ** (attempt + 1) * 2, 20)
                time.sleep(wait_s)
            except Exception as e:
                wait_s = min(2 ** (attempt + 1) * 2, 20)
                time.sleep(wait_s)

        if not model_not_found and model:
            # Explicit model was requested and did not 404/403; do not fall back on transient failures
            break

        CONSECUTIVE_FAILURES += 1
        if CONSECUTIVE_FAILURES >= 5:
            print("FATAL: 5 consecutive Groq failures. Aborting Groq steps loudly!")
            sys.exit(1)

    return {"label": "ordinary", "confidence": 0.0, "cues": ["fallback_exhausted"]}


# Backward-compatible wrapper for existing callers
def classify_fake_type(text, max_retries=5):
    res = classify(text)
    return {
        "label": res["label"],
        "rationale": f"Confidence: {res['confidence']:.2f}. Cues: {', '.join(res.get('cues', []))}"
    }


if __name__ == "__main__":
    # Smoke test on 4 invented sentences; print labels only
    smoke_sentences = [
        "Treasury department announces quarterly refunding estimate of 125 billion dollars",
        "Area dog receives honorary PhD in barking at delivery trucks",
        "10 everyday items you won't believe are secretly ruining your sleep",
        "Leaked documents reveal government is spraying mind-control chemicals in drinking water",
    ]
    for s in smoke_sentences:
        res = classify(s)
        print(res["label"])
