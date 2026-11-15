import os
import time
import json
import requests
import base64
import urllib3
from io import BytesIO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

VALID_VERDICTS = {"TRUE", "FALSE", "UNVERIFIABLE"}
GROQ_STATS = {"attempted": 0, "failed": 0}
LAST_CALL_STATS = {"attempts": 0}


def _build_evidence_block(evidence_results, max_items=5, max_chars=500):
    lines = []
    for i, item in enumerate(evidence_results[:max_items], start=1):
        lines.append(
            f"[{i}] Source: {item.get('title', 'unknown')} ({item.get('url', '')})\n"
            f"    NLI: {item.get('nli_label', 'n/a')} | "
            f"Consistency: {item.get('consistency_verdict', 'n/a')} | "
            f"Semantic similarity: {item.get('semantic_similarity', 0):.2f}\n"
            f"    Text: {item.get('evidence', '')[:max_chars]}"
        )
    return "\n\n".join(lines)


def _call_groq_chat(model, api_key, payload_messages, timeout=30):
    """
    Helper to execute one chat completion call against Groq API.
    Returns (response_or_None, status_code, retry_after_or_None, exception_or_None)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": payload_messages,
        "temperature": 0.2,
        "max_completion_tokens": 400,
        "response_format": {"type": "json_object"},
    }
    GROQ_STATS["attempted"] += 1
    try:
        resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=timeout, verify=False)
        retry_after = resp.headers.get("Retry-After")
        if resp.status_code != 200:
            GROQ_STATS["failed"] += 1
        return resp, resp.status_code, retry_after, None
    except requests.exceptions.Timeout as te:
        GROQ_STATS["failed"] += 1
        return None, 408, None, te
    except Exception as e:
        GROQ_STATS["failed"] += 1
        return None, 500, None, e


def llm_verify_with_status(claim, evidence_results):
    """
    Ask Groq model to verify a claim based on evidence passages.
    Returns (result_or_None, status, model).
    status is one of: 'ok', 'rate_limited', 'error', 'no_key'.
    Retries on 429, 500, 502, 503, timeouts: up to 4 attempts, backoff 2, 4, 8, 16s (honors Retry-After, cap 30s).
    If still failing, tries 1 attempt with GROQ_FAKETYPE_MODEL_2 if set.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, "no_key", None

    if not evidence_results:
        return None, "ok", None

    evidence_block = _build_evidence_block(evidence_results)

    system_prompt = (
        "You are a fact-checking assistant. You will be given a claim and "
        "a set of evidence passages retrieved from the web, each already "
        "scored by an NLI model and an entity-consistency checker. "
        "Decide whether the CLAIM is TRUE, FALSE, or UNVERIFIABLE based "
        "strictly on the evidence given -- do not use outside knowledge "
        "beyond what's in the evidence, and do not guess if the evidence "
        "is thin or off-topic (say UNVERIFIABLE in that case). "
        "Respond ONLY with a JSON object: "
        '{"verdict": "TRUE|FALSE|UNVERIFIABLE", "confidence": 0.0-1.0, '
        '"reasoning": "<=60 words explaining why"}'
    )
    user_prompt = f"Claim: {claim}\n\nEvidence:\n{evidence_block}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    primary_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    fallback_model = os.environ.get("GROQ_FAKETYPE_MODEL_2")

    backoffs = [2, 4, 8, 16]
    last_status = "error"
    active_model = primary_model
    LAST_CALL_STATS["attempts"] = 0

    for attempt in range(4):
        LAST_CALL_STATS["attempts"] = attempt + 1
        resp, code, retry_after, exc = _call_groq_chat(primary_model, api_key, messages, timeout=30)
        if resp is not None and code == 200:
            try:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                    if raw.lower().startswith("json"):
                        raw = raw.split("\n", 1)[-1]
                parsed = json.loads(raw)
                verdict = str(parsed.get("verdict", "")).upper().strip()
                if verdict in VALID_VERDICTS:
                    try:
                        confidence = float(parsed.get("confidence", 0.5))
                    except (TypeError, ValueError):
                        confidence = 0.5
                    confidence = max(0.0, min(1.0, confidence))
                    reasoning = str(parsed.get("reasoning", "")).strip()
                    return {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}, "ok", primary_model
                else:
                    return None, "error", primary_model
            except Exception:
                return None, "error", primary_model

        # Error handling
        if code == 429:
            last_status = "rate_limited"
        elif code in (500, 502, 503, 408) or exc is not None:
            last_status = "error"
        else:
            # Other 4xx error (e.g. 400 Bad Request, 401 Unauthorized)
            last_status = "error"
            break

        if attempt < 3:
            sleep_sec = backoffs[attempt]
            if retry_after:
                try:
                    sleep_sec = float(retry_after)
                except (ValueError, TypeError):
                    pass
            sleep_sec = min(sleep_sec, 30)
            time.sleep(sleep_sec)

    # Try fallback model if configured
    if fallback_model and fallback_model.strip():
        LAST_CALL_STATS["attempts"] = 5
        active_model = fallback_model.strip()
        time.sleep(2)  # short safety pause before fallback attempt
        resp, code, retry_after, exc = _call_groq_chat(active_model, api_key, messages, timeout=30)
        if resp is not None and code == 200:
            try:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                    if raw.lower().startswith("json"):
                        raw = raw.split("\n", 1)[-1]
                parsed = json.loads(raw)
                verdict = str(parsed.get("verdict", "")).upper().strip()
                if verdict in VALID_VERDICTS:
                    try:
                        confidence = float(parsed.get("confidence", 0.5))
                    except (TypeError, ValueError):
                        confidence = 0.5
                    confidence = max(0.0, min(1.0, confidence))
                    reasoning = str(parsed.get("reasoning", "")).strip()
                    return {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}, "ok", active_model
            except Exception:
                pass
        if code == 429:
            last_status = "rate_limited"
        else:
            last_status = "error"

    return None, last_status, active_model


def llm_verify(claim, evidence_results):
    """
    Wrapper preserving signature: returns dict or None.
    """
    res, status, model = llm_verify_with_status(claim, evidence_results)
    return res


def transcribe_audio(audio_bytes, filename="voice_claim.wav", language=None):
    """
    Send recorded audio bytes to Groq Whisper, return transcript text
    or None. Never raises past this function's caller boundary in a
    way that isn't caught -- app.py wraps the call in try/except.
    """
    if not GROQ_API_KEY:
        return None
    if not audio_bytes:
        return None

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    data = {"model": GROQ_WHISPER_MODEL, "response_format": "json"}
    if language:
        data["language"] = language
    files = {"file": (filename, audio_bytes)}

    response = requests.post(
        TRANSCRIBE_URL, headers=headers, data=data, files=files, timeout=60, verify=False
    )
    response.raise_for_status()

    return response.json().get("text", "").strip() or None


def describe_image(pil_image):
    """
    Fallback for when OCR finds little/no text -- i.e. a real photo,
    not a text screenshot. Asks a Groq vision model to describe what's
    actually visible, phrased as a checkable claim, so image-only
    submissions still get something to run through the verifier.
    Returns None on missing key / failure -- never replaces OCR when
    OCR already found real text.
    """
    if not GROQ_API_KEY:
        return None
    if pil_image is None:
        return None

    buffer = BytesIO()
    rgb_image = pil_image.convert("RGB")
    rgb_image.save(buffer, format="JPEG", quality=85)
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe exactly what this image shows, in one "
                            "or two plain factual sentences, phrased as a "
                            "checkable claim (who/what/where/when, if visible). "
                            "Do not guess at anything not visible in the image. "
                            "If the image contains no clear factual claim, just "
                            "describe the scene."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            }
        ],
        "temperature": 0.2,
        "max_completion_tokens": 200,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=30, verify=False)
    response.raise_for_status()

    text = response.json()["choices"][0]["message"]["content"].strip()
    return text or None
