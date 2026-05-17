"""
Web content extractor using httpx + BeautifulSoup4 with a Playwright fallback
for JavaScript-rendered pages.

Flow:
  1. Fast path — httpx fetch + BeautifulSoup parse (< 1 s typical)
  2. If extracted text < 500 chars, try Playwright headless Chromium
  3. Return whichever path produced more content
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag

_MAX_BODY_BYTES = 800 * 1024  # 800 KB
_PW_FALLBACK_THRESHOLD = 500  # chars — below this, escalate to real browser

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Sec-CH-UA": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
}

_CONTENT_SELECTORS = [
    "article",
    "main",
    "[role='main']",
    ".td-post-content",
    ".post-content",
    ".article-body",
    ".article-content",
    ".entry-content",
    ".post-body",
    ".story-body",
    ".content-body",
    ".markdown-body",
    ".prose",
    ".docs-content",
    ".doc-content",
    ".documentation",
    ".content",
    "#content",
    "#main-content",
]

_NOISE_CLASS_FRAGMENTS = [
    "nav", "footer", "header", "sidebar", "menu", "cookie", "banner",
    r"\bad\b", r"\bad-", r"-ad\b",   # "ad" only at word/hyphen boundary
]

# Tags whose classes should never be checked — stripping these would gut the page
_NOISE_CLASS_SKIP_TAGS = frozenset(["html", "body", "article", "main", "section", "div"])

_NOISE_TAGS = [
    "nav", "header", "footer", "script", "style", "noscript",
    "iframe", "form", "button",
]


@dataclass
class ExtractedContent:
    title: str
    text: str
    via_browser: bool = False


# ---------------------------------------------------------------------------
# Playwright shared browser
# ---------------------------------------------------------------------------

_pw_instance = None
_pw_browser = None
_pw_lock: asyncio.Lock | None = None


async def _get_pw_browser():
    """Return a shared Playwright Chromium browser, launching it if needed."""
    global _pw_instance, _pw_browser, _pw_lock
    if _pw_lock is None:
        _pw_lock = asyncio.Lock()
    async with _pw_lock:
        if _pw_browser is None or not _pw_browser.is_connected():
            from playwright.async_api import async_playwright
            _pw_instance = await async_playwright().start()
            _pw_browser = await _pw_instance.chromium.launch(headless=True)
    return _pw_browser


async def _playwright_extract(url: str) -> ExtractedContent | None:
    """Fetch *url* with a real headless browser. Returns None on any failure."""
    try:
        browser = await _get_pw_browser()
        ctx = await browser.new_context(
            user_agent=_BROWSER_HEADERS["User-Agent"],
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        try:
            page = await ctx.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=25_000)
            except Exception:
                # networkidle timeout is common — use whatever loaded so far
                pass
            html = await page.content()
        finally:
            await ctx.close()
        result = parse_html(html, base_url=url)
        result.via_browser = True
        return result
    except Exception as exc:
        print(f"[playwright] failed for {url}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract(url: str) -> ExtractedContent:
    """Fetch *url* and return its readable text.

    Raises:
        ValueError: For 404 or 5xx responses.
        httpx.HTTPError: For network-level failures.
    """
    # --- Fast path: httpx ---------------------------------------------------
    async with httpx.AsyncClient(
        headers=_BROWSER_HEADERS,
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
        max_redirects=10,
    ) as client:
        response = await client.get(url)

    status = response.status_code
    if status == 404:
        raise ValueError(f"Page not found: {url}")
    if status >= 500:
        raise ValueError(f"Server error ({status}): {url}")

    fast = parse_html(response.text, base_url=url)

    if len(fast.text) >= _PW_FALLBACK_THRESHOLD:
        return fast

    # --- Slow path: Playwright headless browser ------------------------------
    print(f"[extractor] thin content ({len(fast.text)} chars), trying browser: {url}", file=sys.stderr)
    pw = await _playwright_extract(url)
    if pw and len(pw.text) > len(fast.text):
        print(f"[extractor] browser got {len(pw.text)} chars", file=sys.stderr)
        return pw

    return fast


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def parse_html(html: str, base_url: str) -> ExtractedContent:
    raw = html
    if len(raw.encode("utf-8", errors="replace")) > _MAX_BODY_BYTES:
        body_start = raw.lower().find("<body")
        raw = raw[body_start: body_start + _MAX_BODY_BYTES] if body_start != -1 else raw[:_MAX_BODY_BYTES]

    soup = BeautifulSoup(raw, "html.parser")

    title = _extract_title(soup, base_url)
    jsonld_text = _extract_jsonld(soup)
    meta_text = _extract_meta(soup)

    _strip_noise(soup)
    text = _extract_content(soup)

    # Supplement with SEO metadata when DOM content is thin
    if len(text) < 150:
        text = max([jsonld_text, meta_text, text], key=len)

    return ExtractedContent(title=title, text=text)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_title(soup: BeautifulSoup, base_url: str) -> str:
    h1 = soup.find("h1")
    if h1 and isinstance(h1, Tag):
        t = h1.get_text(strip=True)
        if t:
            return t
    title_tag = soup.find("title")
    if title_tag and isinstance(title_tag, Tag):
        t = title_tag.get_text(strip=True)
        if t:
            return t
    try:
        return urlparse(base_url).hostname or base_url
    except Exception:
        return base_url


@lru_cache(maxsize=None)
def _noise_patterns():
    return [re.compile(p) for p in _NOISE_CLASS_FRAGMENTS]


def _class_is_noisy(tag: Tag) -> bool:
    if tag.name in _NOISE_CLASS_SKIP_TAGS:
        return False
    classes = tag.get("class", [])
    if isinstance(classes, str):
        classes = [classes]
    joined = " ".join(classes).lower()
    return any(p.search(joined) for p in _noise_patterns())


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for tag in soup.find_all(True):
        if isinstance(tag, Tag) and tag.attrs is not None and _class_is_noisy(tag):
            tag.decompose()


def _extract_content(soup: BeautifulSoup) -> str:
    for selector in _CONTENT_SELECTORS:
        for el in soup.select(selector):
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 100:
                return _clean_text(text)
    body = soup.find("body")
    container = body if body else soup
    text = container.get_text(separator="\n", strip=True) if isinstance(container, Tag) else soup.get_text(separator="\n", strip=True)
    return _clean_text(text)


_MULTI_BLANK = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    return _MULTI_BLANK.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# JSON-LD and meta tag fallbacks
# ---------------------------------------------------------------------------

_LD_FIELDS = ("articleBody", "description", "abstract")
_META_OG = ("og:description", "twitter:description")
_META_NAME = ("description",)


def _extract_jsonld(soup: BeautifulSoup) -> str:
    best = ""
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or ""
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for obj in (data if isinstance(data, list) else [data]):
            if not isinstance(obj, dict):
                continue
            for field in _LD_FIELDS:
                val = obj.get(field, "")
                if isinstance(val, str) and len(val) > len(best):
                    best = val
    return best.strip()


def _extract_meta(soup: BeautifulSoup) -> str:
    for prop in _META_OG:
        tag = soup.find("meta", property=prop)
        if tag and isinstance(tag, Tag):
            val = tag.get("content", "")
            if isinstance(val, str) and val.strip():
                return val.strip()
    for name in _META_NAME:
        tag = soup.find("meta", attrs={"name": name})
        if tag and isinstance(tag, Tag):
            val = tag.get("content", "")
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""
