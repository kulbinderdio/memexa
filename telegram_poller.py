"""
Async Telegram bot poller for memexa-web.

Polls getUpdates every 60 seconds, extracts URLs from messages and
channel posts, and calls on_url_found for each discovered URL.
"""

from __future__ import annotations

import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx

_TELEGRAM_BASE = "https://api.telegram.org/bot"
_LONG_POLL_TIMEOUT = 30   # seconds Telegram holds the connection waiting for updates
_RETRY_DELAY = 5          # seconds to wait after an error before retrying
_UPDATE_ID_FILE = Path.home() / ".memexa-web" / "last_update_id"

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

_STATUS_MESSAGES: dict[int, str] = {
    401: "Bot token invalid (401) — check your Telegram bot token",
    403: "Bot is forbidden (403) — the bot may have been blocked",
    404: "Bot token not recognised (404) — check your Telegram bot token",
    429: "Rate limited by Telegram (429) — backing off",
}


def _log(msg: str) -> None:
    print(f"📡 [Telegram] {msg}", file=sys.stderr, flush=True)


def _err(msg: str) -> None:
    print(f"❌ [Telegram] {msg}", file=sys.stderr, flush=True)


class TelegramPoller:
    """Async Telegram bot long-poller (asyncio task, not a thread)."""

    def __init__(self, on_url_found: Callable[[str], Awaitable[None]]) -> None:
        self._on_url_found = on_url_found
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._token: str = ""
        self._last_update_id: int = self._load_update_id()

        self.is_running: bool = False
        self.last_poll_time: Optional[datetime] = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, token: str) -> None:
        """Start (or restart) the polling loop with *token*."""
        if self._task and not self._task.done():
            await self.stop()
        self._token = token
        self.last_error = None
        self._task = asyncio.create_task(self._run(), name="telegram-poller")
        _log("Poller started.")

    async def stop(self) -> None:
        """Cancel the polling task and wait for it to finish."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.is_running = False
        _log("Poller stopped.")

    async def restart(self, token: str) -> None:
        """Convenience alias for stop + start."""
        await self.stop()
        await self.start(token)

    # ------------------------------------------------------------------
    # Internal polling loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        self.is_running = True
        try:
            while True:
                ok = await self._poll_once()
                if not ok:
                    await asyncio.sleep(_RETRY_DELAY)
                # If ok (even with zero updates), loop immediately —
                # the next long-poll call blocks in Telegram until a message arrives.
        except asyncio.CancelledError:
            pass
        finally:
            self.is_running = False

    async def _poll_once(self) -> bool:
        """Returns True on a clean response (even if no updates), False on error."""
        url = (
            f"{_TELEGRAM_BASE}{self._token}/getUpdates"
            f"?offset={self._last_update_id + 1}&limit=100&timeout={_LONG_POLL_TIMEOUT}"
        )
        try:
            async with httpx.AsyncClient(timeout=_LONG_POLL_TIMEOUT + 10.0) as client:
                resp = await client.get(url)

            self.last_poll_time = datetime.utcnow()

            if resp.status_code != 200:
                msg = _STATUS_MESSAGES.get(
                    resp.status_code,
                    f"HTTP {resp.status_code} from Telegram",
                )
                self.last_error = msg
                _err(msg)
                return False

            data = resp.json()

            if not data.get("ok"):
                desc = data.get("description", "Unknown error")
                self.last_error = desc
                _err(f"Telegram API error: {desc}")
                return False

            self.last_error = None
            updates: list[dict] = data.get("result", [])

            for update in updates:
                update_id: int = update["update_id"]
                if update_id > self._last_update_id:
                    self._last_update_id = update_id
                await self._process_update(update)

            if updates:
                self._save_update_id(self._last_update_id)
                _log(f"Polled {len(updates)} update(s); last_update_id={self._last_update_id}")

            return True

        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            msg = f"Network error: {exc}"
            self.last_error = msg
            _err(msg)
            return False
        except Exception as exc:  # noqa: BLE001
            msg = f"Unexpected error: {exc}"
            self.last_error = msg
            _err(msg)
            return False

    async def _process_update(self, update: dict) -> None:
        """Extract URLs from a Telegram update dict and fire the callback."""
        text: str = ""

        # Regular message
        msg = update.get("message", {})
        if msg and isinstance(msg.get("text"), str):
            text = msg["text"]

        # Channel post
        channel_post = update.get("channel_post", {})
        if channel_post and isinstance(channel_post.get("text"), str):
            text = channel_post["text"]

        if not text:
            return

        urls = _URL_RE.findall(text)
        for raw_url in urls:
            # Strip trailing punctuation that isn't part of the URL
            url = raw_url.rstrip(".,;:!?)")
            _log(f"Found URL: {url}")
            try:
                await self._on_url_found(url)
            except Exception as exc:  # noqa: BLE001
                _err(f"on_url_found raised: {exc}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_update_id(self) -> int:
        try:
            return int(_UPDATE_ID_FILE.read_text().strip())
        except Exception:
            return 0

    def _save_update_id(self, update_id: int) -> None:
        try:
            _UPDATE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
            _UPDATE_ID_FILE.write_text(str(update_id))
        except Exception as exc:
            _err(f"Could not persist last_update_id: {exc}")
