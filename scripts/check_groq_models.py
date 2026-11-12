import os
import sys
import re
import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY set: {bool(groq_key)}")
if not groq_key:
    print("FATAL: GROQ_API_KEY is not set!")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {groq_key}",
    "Content-Type": "application/json",
}

try:
    resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=15, verify=False)
    resp.raise_for_status()
    data = resp.json()
except Exception as e:
    print(f"FATAL: Error calling Groq models endpoint: {e}")
    sys.exit(1)

models = data.get("data", [])

print("--- ALL GROQ MODEL IDS ---")
for m in sorted(models, key=lambda x: x.get("id", "")):
    print(m.get("id", ""))

# Filter: not whisper, vision-only or guard models (also safeguard)
excluded_keywords = ["whisper", "vision", "guard", "safeguard"]
chat_models = []
for m in models:
    mid = m.get("id", "")
    if any(k in mid.lower() for k in excluded_keywords):
        continue
    chat_models.append(mid)

def extract_size(mid):
    # e.g. 120b -> 120, 27b -> 27, 20b -> 20, 7b -> 7
    match = re.search(r"(\d+)b", mid.lower())
    if match:
        return int(match.group(1))
    return 0

ranked = sorted(chat_models, key=lambda m: (extract_size(m), m), reverse=True)

model_1 = os.getenv("GROQ_FAKETYPE_MODEL") or (ranked[0] if len(ranked) > 0 else "")
model_2 = os.getenv("GROQ_FAKETYPE_MODEL_2") or (ranked[1] if len(ranked) > 1 else "")

print("\n--- CHOSEN PAIR ---")
print(f"GROQ_FAKETYPE_MODEL:   {model_1}")
print(f"GROQ_FAKETYPE_MODEL_2: {model_2}")
