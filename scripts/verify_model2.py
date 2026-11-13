import os
import sys
import time
import requests
import urllib3
import pandas as pd
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY set: {bool(groq_key)}")
if not groq_key:
    print("FATAL: GROQ_API_KEY is not set!")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from reasoning.faketype_llm import classify

headers = {
    "Authorization": f"Bearer {groq_key.strip()}",
    "Content-Type": "application/json",
}

# 1. Fetch full models list
resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=15, verify=False)
resp.raise_for_status()
all_models = [m["id"] for m in resp.json().get("data", [])]

print("\n--- FULL GROQ MODELS LIST (IDs ONLY) ---")
for mid in sorted(all_models):
    print(mid)

model_1 = os.getenv("GROQ_FAKETYPE_MODEL", "openai/gpt-oss-120b")
model_2 = os.getenv("GROQ_FAKETYPE_MODEL_2", "qwen/qwen3.8-27b")

print(f"\nGROQ_FAKETYPE_MODEL_2 ({model_2}) in models list: {model_2 in all_models}")
print(f"GROQ_FAKETYPE_MODEL ({model_1}) in models list: {model_1 in all_models}")

# 2. For 5 dev rows, call both models and print the API response's "model" field plus label and cues from each
dev_path = os.path.join(PROJECT_ROOT, "data", "eval", "faketype_dev.csv")
df = pd.read_csv(dev_path)
if "language" in df.columns:
    df = df[df["language"].astype(str).str.lower() == "en"].copy()

sample_df = df.head(5)

print("\n--- 5 DEV ROWS: MODEL 1 vs MODEL 2 RESPONSES ---")
for idx, r in sample_df.iterrows():
    text = r["text"]
    print(f"\nRow {idx+1}: {text[:60]}...")

    res1 = classify(text, model=model_1)
    time.sleep(1.0)
    res2 = classify(text, model=model_2)
    time.sleep(1.0)

    print(f"  Model 1 API 'model': {res1.get('model')} | Label: {res1.get('label')} | Cues: {res1.get('cues')}")
    print(f"  Model 2 API 'model': {res2.get('model')} | Label: {res2.get('label')} | Cues: {res2.get('cues')}")
