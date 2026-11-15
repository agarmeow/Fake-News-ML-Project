import requests
from bs4 import BeautifulSoup

query = "Turkey capital is Ankara"
url = "https://html.duckduckgo.com/html/"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36"}
resp = requests.post(url, data={"q": query}, headers=headers, timeout=15, verify=False)
print("status:", resp.status_code)
print("length:", len(resp.text))
print("text snippet:", resp.text[:400])
