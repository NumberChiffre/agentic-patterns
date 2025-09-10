from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import re


# Common tracking query params to drop during normalization
TRACKING_PREFIXES: tuple[str, ...] = (
    "utm",
    "utm_",
    "ref",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "igshid",
)


def normalize_url(url: str) -> str:
    if not url:
        return url
    url = url.strip()
    sp = urlsplit(url)
    host = sp.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = sp.path.rstrip("/") or "/"
    # Drop tracking params, case-insensitive by key
    params = [
        (k, v)
        for (k, v) in parse_qsl(sp.query, keep_blank_values=True)
        if not any(k.lower().startswith(p) for p in TRACKING_PREFIXES)
    ]
    # Sort for stability
    params.sort()
    query = urlencode(params)
    return urlunsplit((sp.scheme.lower(), host, path, query, ""))


def dedupe_citations(citations: list[dict]) -> list[dict]:
    if not citations:
        return []
    seen: set[str] = set()
    result: list[dict] = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        raw_url = str(c.get("url", "")).strip()
        title = str(c.get("title", "")).strip()
        norm = normalize_url(raw_url)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append({"title": title, "url": norm})
    return result


def merge_and_dedupe(*citation_lists: list[list[dict]] | list[dict]) -> list[dict]:
    merged: list[dict] = []
    for lst in citation_lists:
        if not lst:
            continue
        # Allow both list[dict] and splatted lists
        if isinstance(lst, list):
            merged.extend([c for c in lst if isinstance(c, dict)])
    return dedupe_citations(merged)


_MD_LINK_RE = re.compile(r"\[([^\]]{1,256})\]\((https?://[^)\s]+)\)")
_BARE_URL_RE = re.compile(r"(?P<url>https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)")


def extract_citations_from_text(text: str) -> list[dict]:
    if not text:
        return []
    citations: list[dict] = []
    # Markdown links
    for m in _MD_LINK_RE.finditer(text):
        title = (m.group(1) or "").strip()
        url = (m.group(2) or "").strip()
        if url:
            citations.append({"title": title, "url": url})
    # Bare URLs (avoid duplicates with previous by final dedupe)
    for m in _BARE_URL_RE.finditer(text):
        url = (m.group("url") or "").strip()
        if url:
            # Title fallback: hostname
            sp = urlsplit(url)
            host = sp.netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            citations.append({"title": host or "", "url": url})
    return dedupe_citations(citations)


def _domain_from_url(url: str) -> str:
    sp = urlsplit(url)
    host = sp.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def clean_citations_for_export(citations: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    seen: set[str] = set()
    for c in citations or []:
        url = normalize_url(str(c.get("url", "")).strip())
        if not url or url in seen:
            continue
        seen.add(url)
        cleaned.append({"title": _domain_from_url(url), "url": url})
    return cleaned


