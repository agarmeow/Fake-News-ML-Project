import os
import json
import requests
import base64
from io import BytesIO

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
GROQ_VISION_MODEL = os.environ.get(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
)

CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

VALID_VERDICTS = {"TRUE", "FALSE", "UNVERIFIABLE"}


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


def llm_verify(claim, evidence_results):
    """
    Ask Groq's gpt-oss model to read the SAME evidence the rule-based
    pipeline already retrieved (no new search calls) and give an
    independent verdict + plain-English reasoning.

    Returns None if the key is missing, evidence is empty, or the
    call/parse fails -- caller treats None as "no LLM opinion".
    Deliberately does NOT swallow exceptions itself; the caller in
    fact_checker.py wraps this in try/except so a Groq outage never
    breaks the working pipeline.
    """
    if not GROQ_API_KEY:
        return None
    if not evidence_results:
        return None

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

    payload = {
        "model": GROQ_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_completion_tokens": 400,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw.split("\n", 1)[-1]

    parsed = json.loads(raw)

    verdict = str(parsed.get("verdict", "")).upper().strip()
    if verdict not in VALID_VERDICTS:
        return None

    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(parsed.get("reasoning", "")).strip()

    return {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}


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
        TRANSCRIBE_URL, headers=headers, data=data, files=files, timeout=60
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

    response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    text = response.json()["choices"][0]["message"]["content"].strip()
    return text or None
