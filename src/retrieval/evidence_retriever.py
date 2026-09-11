import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


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
}

REFERENCE_DOMAINS = {
    "wikipedia.org",
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
        if domain == trusted_domain or domain.endswith(
            "." + trusted_domain
        ):
            return 1.0, "High"

    for reputable_domain in REPUTABLE_DOMAINS:
        if domain == reputable_domain or domain.endswith(
            "." + reputable_domain
        ):
            return 0.85, "High"

    for reference_domain in REFERENCE_DOMAINS:
        if domain == reference_domain or domain.endswith(
            "." + reference_domain
        ):
            return 0.70, "Medium"

    return 0.50, "Unknown"


# ============================================================
# Web search
# ============================================================

def search_web(query, max_results=5):
    """Search the web and return candidate sources."""

    url = "https://html.duckduckgo.com/html/"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        )
    }

    response = requests.post(
        url,
        data={"q": query},
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

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

    return results


# ============================================================
# Page extraction
# ============================================================

def fetch_page_text(url, max_chars=5000):
    """Fetch readable text from a webpage."""

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
            timeout=15
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

        return text[:max_chars]

    except requests.RequestException as e:

        print(f"Could not fetch page: {e}")
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

    print(f"\nSearching for: {claim}")

    search_results = search_web(
        claim,
        max_results=max_results
    )

    evidence = []

    for result in search_results:

        score, quality = get_source_quality(
            result["url"]
        )

        page_text = fetch_page_text(
            result["url"]
        )

        evidence.append({
            "title": result["title"],
            "url": result["url"],
            "snippet": result["snippet"],
            "text": page_text,
            "source_quality": quality,
            "source_quality_score": score
        })

    # Rank by source quality
    evidence.sort(
        key=lambda x: x["source_quality_score"],
        reverse=True
    )

    return evidence


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("COMPLETE EVIDENCE RETRIEVAL TEST")
    print("=" * 60)

    claim = "India's capital city is New Delhi."

    evidence = retrieve_evidence(
        claim,
        max_results=5
    )

    print("\n" + "=" * 60)
    print("RANKED EVIDENCE")
    print("=" * 60)

    for i, item in enumerate(evidence, start=1):

        print(f"\nSOURCE {i}")
        print("-" * 60)

        print(f"Title: {item['title']}")
        print(f"Domain: {get_domain(item['url'])}")
        print(f"Quality: {item['source_quality']}")
        print(
            f"Quality Score: "
            f"{item['source_quality_score']:.2f}"
        )

        print(f"URL: {item['url']}")

        print(
            f"\nSnippet: "
            f"{item['snippet']}"
        )

        if item["text"]:

            print("\nEvidence Text:")
            print(item["text"][:500])

        else:

            print("\nEvidence Text: FAILED")