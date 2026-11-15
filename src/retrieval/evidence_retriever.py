import os
import re
import time
import json
import random
import hashlib
import requests
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# Cache configuration
# ============================================================

CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "cache")
)
SEARCH_CACHE_DIR = os.path.join(CACHE_DIR, "search")
PAGES_CACHE_DIR = os.path.join(CACHE_DIR, "pages")

CACHE_STATS = {
    "search_hit": 0,
    "search_miss": 0,
    "page_hit": 0,
    "page_miss": 0,
}

TIMING_STATS = {
    "search_web": 0.0,
    "page_fetch": 0.0,
}

RETRIEVAL_METADATA = {
    "retrieval_source": "cache",
    "stale_cache": False,
}


def reset_stats():
    CACHE_STATS["search_hit"] = 0
    CACHE_STATS["search_miss"] = 0
    CACHE_STATS["page_hit"] = 0
    CACHE_STATS["page_miss"] = 0
    TIMING_STATS["search_web"] = 0.0
    TIMING_STATS["page_fetch"] = 0.0


def reset_retrieval_metadata():
    RETRIEVAL_METADATA["retrieval_source"] = "cache"
    RETRIEVAL_METADATA["stale_cache"] = False


def get_retrieval_metadata():
    return dict(RETRIEVAL_METADATA)


def _get_cache_ttl():
    try:
        hours = float(os.environ.get("CACHE_TTL_HOURS", "720"))
        return hours * 3600.0
    except Exception:
        return 720.0 * 3600.0


def _is_cache_enabled():
    return os.environ.get("CACHE_DISABLE") != "1"


def _is_cache_only():
    return os.environ.get("CACHE_ONLY") == "1"


def _get_cache_path(directory, key, ext):
    os.makedirs(directory, exist_ok=True)
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(directory, f"{h}.{ext}")


def _load_cache(file_path, allow_stale=False):
    if not _is_cache_enabled():
        return None
    if os.path.exists(file_path):
        try:
            mtime = os.path.getmtime(file_path)
            is_fresh = (time.time() - mtime) < _get_cache_ttl()
            if is_fresh or allow_stale or _is_cache_only():
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    is_stale = not is_fresh
                    return content, is_stale
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

def _simplify_query(query):
    # remove possessive 's or ’s
    q = re.sub(r"['’]s\b", "", query, flags=re.IGNORECASE)
    # strip punctuation and quotes
    q = re.sub(r'["\'.,?!;:()[\]{}]', ' ', q)
    # collapse whitespace
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def _search_ddg_html(q_str, max_results=5):
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        )
    }
    for attempt in range(2):
        try:
            response = requests.post(
                url,
                data={"q": q_str},
                headers=headers,
                timeout=(5, 15),
                verify=False
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
                    return results
        except Exception as e:
            if os.environ.get("FC_TIMING") == "1":
                print(f"ddg_html attempt {attempt + 1} failed: {e}")

        if attempt < 1:
            time.sleep(random.uniform(0.5, 1.5))
    return []


def _search_ddg_lite(q_str, max_results=5):
    url = "https://lite.duckduckgo.com/lite/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        )
    }
    for attempt in range(2):
        try:
            response = requests.post(
                url,
                data={"q": q_str},
                headers=headers,
                timeout=(5, 15),
                verify=False
            )
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                links = soup.select(".result-link")
                snippets = soup.select(".result-snippet")
                for i in range(min(len(links), max_results)):
                    link_el = links[i]
                    href = link_el.get("href", "")
                    title = link_el.get_text(" ", strip=True)
                    snip = snippets[i].get_text(" ", strip=True) if i < len(snippets) else ""
                    if href:
                        results.append({
                            "title": title,
                            "url": href,
                            "snippet": snip
                        })
                if results:
                    return results
        except Exception as e:
            if os.environ.get("FC_TIMING") == "1":
                print(f"ddg_lite attempt {attempt + 1} failed: {e}")

        if attempt < 1:
            time.sleep(random.uniform(0.5, 1.5))
    return []


def _search_wikipedia(q_str, max_results=5):
    api_url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "FakeNewsFactChecker/1.0 (contact@example.org)"}
    params = {
        "action": "query",
        "list": "search",
        "srsearch": q_str,
        "format": "json",
        "srlimit": max_results
    }
    for attempt in range(2):
        try:
            response = requests.get(
                api_url,
                params=params,
                headers=headers,
                timeout=(5, 15),
                verify=False
            )
            if response.status_code == 200:
                data = response.json()
                search_items = data.get("query", {}).get("search", [])
                if not search_items:
                    return []

                pageids = [str(item["pageid"]) for item in search_items]
                extracts = {}
                if pageids:
                    time.sleep(random.uniform(0.5, 1.5))
                    try:
                        ext_params = {
                            "action": "query",
                            "prop": "extracts",
                            "pageids": "|".join(pageids),
                            "exintro": True,
                            "explaintext": True,
                            "format": "json"
                        }
                        ext_resp = requests.get(
                            api_url,
                            params=ext_params,
                            headers=headers,
                            timeout=(5, 15),
                            verify=False
                        )
                        if ext_resp.status_code == 200:
                            pages = ext_resp.json().get("query", {}).get("pages", {})
                            for pid, pdata in pages.items():
                                extracts[pid] = pdata.get("extract", "")
                    except Exception:
                        pass

                results = []
                for item in search_items:
                    pid = str(item["pageid"])
                    title = item["title"]
                    snip = extracts.get(pid, "")
                    if not snip:
                        raw_snip = item.get("snippet", "")
                        snip = BeautifulSoup(raw_snip, "html.parser").get_text(strip=True)
                    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
                    results.append({
                        "title": f"{title} - Wikipedia",
                        "url": url,
                        "snippet": snip[:1000]
                    })
                if results:
                    return results
        except Exception as e:
            if os.environ.get("FC_TIMING") == "1":
                print(f"wikipedia attempt {attempt + 1} failed: {e}")

        if attempt < 1:
            time.sleep(random.uniform(0.5, 1.5))
    return []


def _execute_search_query(q_str, max_results=5):
    cache_key = f"{q_str.strip().lower()}_{max_results}"
    cache_file = _get_cache_path(SEARCH_CACHE_DIR, cache_key, "json")
    loaded_cache = _load_cache(cache_file, allow_stale=False)
    if not loaded_cache:
        legacy_file = _get_cache_path(SEARCH_CACHE_DIR, f"{q_str}_{max_results}", "json")
        loaded_cache = _load_cache(legacy_file, allow_stale=False)

    if loaded_cache:
        cached_data, is_stale = loaded_cache
        try:
            loaded = json.loads(cached_data)
            # treat existing empty cache entries as misses
            if isinstance(loaded, list) and len(loaded) > 0:
                CACHE_STATS["search_hit"] += 1
                if is_stale:
                    RETRIEVAL_METADATA["stale_cache"] = True
                    RETRIEVAL_METADATA["retrieval_source"] = "stale_cache"
                else:
                    RETRIEVAL_METADATA["retrieval_source"] = "cache"
                if os.environ.get("FC_TIMING") == "1":
                    print(f"[{'STALE ' if is_stale else ''}CACHE HIT] search_web: {q_str[:50]}")
                return loaded
        except Exception:
            pass

    CACHE_STATS["search_miss"] += 1
    if os.environ.get("FC_TIMING") == "1":
        print(f"[CACHE MISS] search_web: {q_str[:50]}")

    if _is_cache_only():
        RETRIEVAL_METADATA["retrieval_source"] = "cache_miss"
        return []

    # 1. Primary: html.duckduckgo.com
    results = _search_ddg_html(q_str, max_results=max_results)
    if results and len(results) > 0:
        RETRIEVAL_METADATA["retrieval_source"] = "ddg_html"
        _save_cache(cache_file, json.dumps(results))
        return results

    time.sleep(random.uniform(0.5, 1.5))

    # 2. Fallback 1: lite.duckduckgo.com
    results = _search_ddg_lite(q_str, max_results=max_results)
    if results and len(results) > 0:
        RETRIEVAL_METADATA["retrieval_source"] = "ddg_lite"
        _save_cache(cache_file, json.dumps(results))
        return results

    time.sleep(random.uniform(0.5, 1.5))

    # 3. Fallback 2: Wikipedia API
    results = _search_wikipedia(q_str, max_results=max_results)
    if results and len(results) > 0:
        RETRIEVAL_METADATA["retrieval_source"] = "wikipedia"
        _save_cache(cache_file, json.dumps(results))
        return results

    # 4. Stale-if-error: if live search fails or returns 0 results and a stale cache entry exists, use it
    stale_cache = _load_cache(cache_file, allow_stale=True)
    if not stale_cache:
        legacy_file = _get_cache_path(SEARCH_CACHE_DIR, f"{q_str}_{max_results}", "json")
        stale_cache = _load_cache(legacy_file, allow_stale=True)

    if stale_cache:
        cached_data, _ = stale_cache
        try:
            loaded = json.loads(cached_data)
            if isinstance(loaded, list) and len(loaded) > 0:
                RETRIEVAL_METADATA["stale_cache"] = True
                RETRIEVAL_METADATA["retrieval_source"] = "stale_cache"
                if os.environ.get("FC_TIMING") == "1":
                    print(f"[STALE-IF-ERROR HIT] search_web: {q_str[:50]}")
                return loaded
        except Exception:
            pass

    RETRIEVAL_METADATA["retrieval_source"] = "cache_miss"
    return []


def search_web(query, max_results=5):
    """Search the web and return candidate sources."""
    t0 = time.time()
    query = str(query)[:200]

    results = _execute_search_query(query, max_results=max_results)

    # (2) if search_web returns 0 results, retry once with a simplified query
    if not results:
        simplified = _simplify_query(query)
        if simplified and simplified.lower() != query.strip().lower():
            if os.environ.get("FC_TIMING") == "1":
                print(f"[RETRY SIMPLIFIED] {query[:40]} -> {simplified[:40]}")
            retry_results = _execute_search_query(simplified, max_results=max_results)
            if retry_results:
                # Add result["retrieval_retry"]=True only when used
                for r in retry_results:
                    r["retrieval_retry"] = True
                results = retry_results

    TIMING_STATS["search_web"] += (time.time() - t0)
    return results


# ============================================================
# Page extraction
# ============================================================

def fetch_page_text(url, max_chars=5000):
    """Fetch readable text from a webpage."""
    if not url:
        return ""

    url_clean = url.strip()
    cache_file = _get_cache_path(PAGES_CACHE_DIR, url_clean, "txt")
    loaded_cache = _load_cache(cache_file, allow_stale=False)
    if loaded_cache:
        cached_text, is_stale = loaded_cache
        if cached_text.strip():
            CACHE_STATS["page_hit"] += 1
            if is_stale:
                RETRIEVAL_METADATA["stale_cache"] = True
            if os.environ.get("FC_TIMING") == "1":
                print(f"[{'STALE ' if is_stale else ''}CACHE HIT] page: {url_clean[:50]}")
            return cached_text[:max_chars]

    CACHE_STATS["page_miss"] += 1
    if os.environ.get("FC_TIMING") == "1":
        print(f"[CACHE MISS] page: {url_clean[:50]}")

    if _is_cache_only():
        return ""

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
            timeout=8,
            verify=False
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

    # Stale-if-error for page fetch
    stale_cache = _load_cache(cache_file, allow_stale=True)
    if stale_cache:
        cached_text, _ = stale_cache
        if cached_text.strip():
            RETRIEVAL_METADATA["stale_cache"] = True
            if os.environ.get("FC_TIMING") == "1":
                print(f"[STALE-IF-ERROR HIT] page: {url_clean[:50]}")
            return cached_text[:max_chars]

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
        item = {
            "title": result["title"],
            "url": result["url"],
            "snippet": result["snippet"],
            "text": final_text,
            "source_quality": quality,
            "source_quality_score": score
        }
        if result.get("retrieval_retry"):
            item["retrieval_retry"] = True
        return item

    t_fetch = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        evidence = list(executor.map(_process_result, search_results))
    TIMING_STATS["page_fetch"] += (time.time() - t_fetch)

    # Rank by source quality
    evidence.sort(
        key=lambda x: x["source_quality_score"],
        reverse=True
    )

    return evidence
