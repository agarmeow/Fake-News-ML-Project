import re

WHITESPACE_PATTERN = re.compile(r"\s+")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def prep_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = WHITESPACE_PATTERN.sub(" ", s).strip()
    t = t.lower()
    t = YEAR_PATTERN.sub("year", t)
    return t
