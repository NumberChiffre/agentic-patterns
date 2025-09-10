from __future__ import annotations

import pytest

from src.citations import normalize_url, dedupe_citations, merge_and_dedupe, extract_citations_from_text


def test_normalize_url_basic_rules() -> None:
    u1 = "HTTPS://WWW.Example.com/Path/To/Doc/?utm_source=x&ref=abc&b=2&a=1#frag"
    u2 = "https://example.com/Path/To/Doc?a=1&b=2"
    assert normalize_url(u1) == u2


def test_normalize_url_trailing_slash_and_root() -> None:
    assert normalize_url("https://example.com/") == "https://example.com/"
    assert normalize_url("https://example.com////") == "https://example.com/"
    assert normalize_url("https://example.com/path/") == "https://example.com/path"


def test_dedupe_citations_by_normalized_url() -> None:
    citations = [
        {"title": "Doc A", "url": "https://www.example.com/x?utm_campaign=y#z"},
        {"title": "Doc A duplicate", "url": "https://example.com/x"},
        {"title": "Doc B", "url": "https://example.com/y?b=2&a=1"},
        {"title": "Doc B dup", "url": "https://example.com/y?a=1&b=2#frag"},
    ]
    deduped = dedupe_citations(citations)
    assert len(deduped) == 2
    assert deduped[0]["title"].startswith("Doc A")
    assert deduped[0]["url"] == "https://example.com/x"
    assert deduped[1]["url"] == "https://example.com/y?a=1&b=2"


def test_merge_and_dedupe_preserves_first_occurrence() -> None:
    a = [{"title": "T1", "url": "https://example.com/a?utm=x"}]
    b = [{"title": "T1 dup", "url": "https://example.com/a"}]
    c = [{"title": "T2", "url": "https://example.com/b"}]
    merged = merge_and_dedupe(a, b, c)
    assert [m["title"] for m in merged] == ["T1", "T2"]


def test_extract_citations_from_text_markdown_and_bare_urls() -> None:
    text = "See [Great Paper](https://www.example.com/p?a=1&utm_source=x) and also https://example.com/p?a=1#frag"
    cites = extract_citations_from_text(text)
    assert len(cites) == 1
    assert cites[0]["url"] == "https://example.com/p?a=1"


