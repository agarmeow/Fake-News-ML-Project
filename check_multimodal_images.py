import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

FILE = "data/multimodal_clean/train.tsv"

df = pd.read_csv(FILE, sep="\t")

urls = df["image_url"].head(100).tolist()

def check_url(url):
    try:
        r = requests.get(
            url,
            timeout=10,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        return r.status_code == 200 and "image" in r.headers.get("Content-Type", "")
    except Exception:
        return False

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(check_url, urls))

working = sum(results)

print(f"Checked: {len(urls)}")
print(f"Working image URLs: {working}")
print(f"Failed image URLs: {len(urls) - working}")