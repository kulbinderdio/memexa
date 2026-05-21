"""
Web content extractor — trafilatura for content quality, Playwright for JS pages.

Flow:
  1. httpx fetch → trafilatura extract
  2. If text < 500 chars, escalate to Playwright headless Chromium
  3. Playwright dismisses common popups before extraction
  4. Both paths use trafilatura; BeautifulSoup only for title fallback
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import trafilatura
import trafilatura.settings
from bs4 import BeautifulSoup, Tag

_MAX_BODY_BYTES = 800 * 1024
_PW_FALLBACK_THRESHOLD = 500

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

# Selectors for popup close/accept buttons to dismiss before extracting
_POPUP_DISMISS_SELECTORS = [
    # Cookie consent
    'button:has-text("Accept all")',
    'button:has-text("Accept cookies")',
    'button:has-text("Accept")',
    'button:has-text("I agree")',
    'button:has-text("Got it")',
    'button:has-text("OK")',
    'button:has-text("Agree")',
    # Generic close buttons
    '[aria-label="Close"]',
    '[aria-label="Dismiss"]',
    '.cookie-accept',
    '#accept-cookies',
    '#cookie-accept',
    '.js-accept-cookies',
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
                pass  # use whatever loaded

            # Dismiss common popups before extracting
            for selector in _POPUP_DISMISS_SELECTORS:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=500):
                        await btn.click(timeout=500)
                        await page.wait_for_timeout(400)
                        break
                except Exception:
                    continue

            html = await page.content()
        finally:
            await ctx.close()

        result = _trafilatura_parse(html, url)
        if result:
            result.via_browser = True
        return result
    except Exception as exc:
        print(f"[playwright] failed for {url}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract(url: str) -> ExtractedContent:
    async with httpx.AsyncClient(
        headers=_BROWSER_HEADERS,
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
        max_redirects=10,
    ) as client:
        response = await client.get(url)

    if response.status_code == 404:
        raise ValueError(f"Page not found: {url}")
    if response.status_code >= 500:
        raise ValueError(f"Server error ({response.status_code}): {url}")

    fast = _trafilatura_parse(response.text, url)

    if fast and len(fast.text) >= _PW_FALLBACK_THRESHOLD:
        return fast

    fast_len = len(fast.text) if fast else 0
    print(f"[extractor] thin content ({fast_len} chars), trying browser: {url}", file=sys.stderr)

    pw = await _playwright_extract(url)
    if pw and len(pw.text) > fast_len:
        print(f"[extractor] browser got {len(pw.text)} chars", file=sys.stderr)
        return pw

    return fast or ExtractedContent(title=_bs_title(response.text, url), text="")


# Keep parse_html as a public entry point (used by Playwright path in tests)
def parse_html(html: str, base_url: str) -> ExtractedContent:
    result = _trafilatura_parse(html, base_url)
    return result or ExtractedContent(title=_bs_title(html, base_url), text="")


# ---------------------------------------------------------------------------
# Trafilatura extraction
# ---------------------------------------------------------------------------

def _trafilatura_parse(html: str, url: str) -> ExtractedContent | None:
    if len(html.encode("utf-8", errors="replace")) > _MAX_BODY_BYTES:
        html = html[:_MAX_BODY_BYTES]

    # bare_extraction returns a dict with text, title, author, etc.
    data = trafilatura.bare_extraction(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        with_metadata=True,
    )

    if data:
        text = (data.get("text") or "").strip()
        title = (data.get("title") or "").strip() or _bs_title(html, url)
        if text:
            return ExtractedContent(title=title, text=text)

    # trafilatura found nothing — try a simpler extract() call as fallback
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        favor_recall=True,
    )
    if text:
        return ExtractedContent(title=_bs_title(html, url), text=text.strip())

    return None


# ---------------------------------------------------------------------------
# Title fallback (BeautifulSoup, used when trafilatura metadata is absent)
# ---------------------------------------------------------------------------

def _bs_title(html: str, url: str) -> str:
    try:
        soup = BeautifulSoup(html[:200_000], "html.parser")
        h1 = soup.find("h1")
        if h1 and isinstance(h1, Tag):
            t = h1.get_text(strip=True)
            if t:
                return t
        tag = soup.find("title")
        if tag and isinstance(tag, Tag):
            t = tag.get_text(strip=True)
            if t:
                return t
    except Exception:
        pass
    try:
        return urlparse(url).hostname or url
    except Exception:
        return url
