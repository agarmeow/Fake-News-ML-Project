from urllib.parse import urlparse


# ============================================================
# Source quality rules
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
    """Extract the domain name from a URL."""

    try:
        domain = urlparse(url).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return ""


def get_source_quality(url):
    """
    Assign a source-quality score.

    Higher score = generally more authoritative.
    """

    domain = get_domain(url)

    # Government / international organizations
    for trusted_domain in HIGH_TRUST_DOMAINS:
        if domain == trusted_domain or domain.endswith(
            "." + trusted_domain
        ):
            return 1.0, "High"

    # Reputable reference/news/research sources
    for reputable_domain in REPUTABLE_DOMAINS:
        if domain == reputable_domain or domain.endswith(
            "." + reputable_domain
        ):
            return 0.85, "High"

    # Reference sources
    for reference_domain in REFERENCE_DOMAINS:
        if domain == reference_domain or domain.endswith(
            "." + reference_domain
        ):
            return 0.70, "Medium"

    # Unknown/general source
    return 0.50, "Unknown"


def rank_sources(results):
    """
    Add source-quality information and rank results.
    """

    ranked_results = []

    for result in results:

        score, quality = get_source_quality(
            result["url"]
        )

        result_copy = result.copy()

        result_copy["source_quality_score"] = score
        result_copy["source_quality"] = quality

        ranked_results.append(result_copy)

    ranked_results.sort(
        key=lambda x: x["source_quality_score"],
        reverse=True
    )

    return ranked_results


if __name__ == "__main__":

    print("=" * 60)
    print("SOURCE QUALITY RANKING TEST")
    print("=" * 60)

    test_sources = [
        {
            "title": "Government Source",
            "url": "https://www.india.gov.in/example",
            "snippet": "Official government information."
        },
        {
            "title": "Britannica",
            "url": "https://www.britannica.com/place/New-Delhi",
            "snippet": "Reference information."
        },
        {
            "title": "Wikipedia",
            "url": "https://en.wikipedia.org/wiki/New_Delhi",
            "snippet": "Reference information."
        },
        {
            "title": "Unknown Website",
            "url": "https://example.com/article",
            "snippet": "Some information."
        }
    ]

    ranked = rank_sources(test_sources)

    print("\nRANKED SOURCES:")
    print("-" * 60)

    for i, result in enumerate(ranked, start=1):

        print(f"\n{i}. {result['title']}")
        print(f"Domain: {get_domain(result['url'])}")
        print(f"Quality: {result['source_quality']}")
        print(
            f"Score: {result['source_quality_score']:.2f}"
        )