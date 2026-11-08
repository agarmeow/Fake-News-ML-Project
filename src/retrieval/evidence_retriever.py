import os
import time
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# Cache configuration
# ============================================================

CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "cache")
)
SEARCH_CACHE_DIR = os.path.join(CACHE_DIR, "search")
PAGES_CACHE_DIR = os.path.join(CACHE_DIR, "pages")
CACHE_TTL = 24 * 3600  # 24 hours


def _is_cache_enabled():
    return os.environ.get("CACHE_DISABLE") != "1"


def _get_cache_path(directory, key, ext):
    os.makedirs(directory, exist_ok=True)
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(directory, f"{h}.{ext}")


def _load_cache(file_path):
    if not _is_cache_enabled():
        return None
    if os.path.exists(file_path):
        mtime = os.path.getmtime(file_path)
        if time.time() - mtime < CACHE_TTL:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return None
    return None


def _save_cache(file_path, content):
    if not _is_cache_enabled():
        return
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass


# ============================================================
# Source quality
# ============================================================

HIGH_TRUST_DOMAINS = {
    "gov.in",
    "nic.in",
    "india.gov.in",
    "who.int",
    "un.org",
    "worldbank.org",
    "imf.org",
    "nasa.gov",
    "isro.gov.in",
}

REPUTABLE_DOMAINS = {
    "britannica.com",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nature.com",
    "sciencedirect.com",
    "nbcnews.com",
    "cbsnews.com",
    "cnn.com",
    "abcnews.go.com",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "usatoday.com",
    "thehindu.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "ndtv.com",
    "pib.gov.in",
    "altnews.in",
    "boomlive.in",
    "variety.com",
}

REFERENCE_DOMAINS = {
    "wikipedia.org",
    "people.com",
}


def get_domain(url):
    """Extract domain from URL."""
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def get_source_quality(url):
    """Assign a source-quality score."""
    domain = get_domain(url)

    for trusted_domain in HIGH_TRUST_DOMAINS:
        if domain == trusted_domain or domain.endswith("." + trusted_domain):
            return 1.0, "High"

    for reputable_domain in REPUTABLE_DOMAINS:
        if domain == reputable_domain or domain.endswith("." + reputable_domain):
            return 0.85, "High"

    for reference_domain in REFERENCE_DOMAINS:
        if domain == reference_domain or domain.endswith("." + reference_domain):
            return 0.70, "Medium"

    return 0.50, "Unknown"


# ============================================================
# Web search
# ============================================================

def search_web(query, max_results=5):
    """Search the web and return candidate sources."""
    query = str(query)[:200]

    cache_file = _get_cache_path(SEARCH_CACHE_DIR, f"{query}_{max_results}", "json")
    cached_data = _load_cache(cache_file)
    if cached_data:
        try:
            return json.loads(cached_data)
        except Exception:
            pass

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        )
    }

    results = []
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                data={"q": query},
                headers=headers,
                timeout=15
            )
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                for result in soup.select(".result")[:max_results]:
                    link_element = result.select_one(".result__a")
                    title_element = result.select_one(".result__title")
                    snippet_element = result.select_one(".result__snippet")

                    if not link_element:
                        continue

                    results.append({
                        "title": (
                            title_element.get_text(" ", strip=True)
                            if title_element else ""
                        ),
                        "url": link_element.get("href", ""),
                        "snippet": (
                            snippet_element.get_text(" ", strip=True)
                            if snippet_element else ""
                        )
                    })

                if results:
                    break
        except Exception as e:
            print(f"search_web attempt {attempt + 1} failed: {e}")

        if attempt < max_retries:
            time.sleep(1.0 * (attempt + 1))

    if results:
        _save_cache(cache_file, json.dumps(results))

    return results


# ============================================================
# Page extraction
# ============================================================

def fetch_page_text(url, max_chars=5000):
    """Fetch readable text from a webpage."""
    if not url:
        return ""

    cache_file = _get_cache_path(PAGES_CACHE_DIR, url, "txt")
    cached_text = _load_cache(cache_file)
    if cached_text is not None:
        return cached_text[:max_chars]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=8
        )
        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for element in soup(
            ["script", "style", "nav", "footer", "header"]
        ):
            element.decompose()

        text = soup.get_text(
            " ",
            strip=True
        )
        text = " ".join(text.split())
        extracted = text[:max_chars]
        if extracted.strip():
            _save_cache(cache_file, extracted)
        return extracted

    except Exception as e:
        print(f"Could not fetch page {url}: {e}")
        return ""


# ============================================================
# Evidence retrieval
# ============================================================

def retrieve_evidence(
    claim,
    max_results=5
):
    """
    Search for a claim, fetch page text,
    and assign source-quality scores.
    """
    claim = str(claim)[:200]
    print(f"\nSearching for: {claim}")

    search_results = search_web(
        claim,
        max_results=max_results
    )

    def _process_result(result):
        score, quality = get_source_quality(result["url"])
        page_text = fetch_page_text(result["url"])
        final_text = page_text.strip() if page_text and page_text.strip() else result.get("snippet", "")
        return {
            "title": result["title"],
            "url": result["url"],
            "snippet": result["snippet"],
            "text": final_text,
            "source_quality": quality,
            "source_quality_score": score
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        evidence = list(executor.map(_process_result, search_results))

    # Rank by source quality
    evidence.sort(
        key=lambda x: x["source_quality_score"],
        reverse=True
    )

    return evidence
