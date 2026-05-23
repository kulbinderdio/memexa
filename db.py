"""
Async SQLite database layer for memexa-web.
Database stored at ~/.memexa-web/data.db
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import aiosqlite

_DB_DIR = Path.home() / ".memexa-web"
_DB_PATH = _DB_DIR / "data.db"

_DEFAULT_SETTINGS: dict[str, str] = {
    "llm_provider": "ollama",
    "ollama_base_url": "http://localhost:11434",
    "ollama_chat_model": "gemma4:latest",
    "ollama_embed_model": "mxbai-embed-large",
    "openai_api_key": "",
    "openai_model": "gpt-4o-mini",
    "claude_api_key": "",
    "claude_model": "claude-sonnet-4-6",
    "telegram_bot_token": "",
    "server_port": "7700",
    "archive_fallback": "false",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS weekly_digests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start  DATE NOT NULL UNIQUE,
    week_end    DATE NOT NULL,
    summary     TEXT,
    item_count  INTEGER,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS items (
    id          TEXT PRIMARY KEY,
    url         TEXT UNIQUE NOT NULL,
    title       TEXT,
    summary     TEXT,
    content     TEXT,
    tags_json   TEXT    DEFAULT '[]',
    embedding_data BLOB DEFAULT x'',
    status      TEXT    DEFAULT 'unread',
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
    url           TEXT,
    source        TEXT DEFAULT 'manual',
    status        TEXT,
    title         TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _db_path() -> str:
    return str(_DB_PATH)


def pack_embedding(vec: list[float]) -> bytes:
    """Encode a list of floats as raw float32 bytes."""
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_embedding(data: bytes) -> list[float]:
    """Decode raw float32 bytes back to a list of floats."""
    if not data:
        return []
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data[: count * 4]))


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return dict(row)


async def init_db() -> None:
    """Create tables and insert default settings if they don't exist."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        await db.executescript(_SCHEMA)
        # Insert defaults only for missing keys
        for key, value in _DEFAULT_SETTINGS.items():
            await db.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        await db.commit()


async def get_settings() -> dict[str, str]:
    """Return all settings as a plain dict."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT key, value FROM settings") as cur:
            rows = await cur.fetchall()
    return {row["key"]: row["value"] for row in rows}


async def update_setting(key: str, value: str) -> None:
    """Upsert a single setting."""
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        await db.commit()


async def fetch_all_items() -> list[dict]:
    """Return all items ordered newest first, excluding embedding_data."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, url, title, summary, tags_json, status, created_at "
            "FROM items ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def fetch_item(id: str) -> dict | None:
    """Return a single item with content, excluding embedding_data."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, url, title, summary, content, tags_json, status, created_at "
            "FROM items WHERE id=?",
            (id,),
        ) as cur:
            row = await cur.fetchone()
    return _row_to_dict(row) if row else None


async def fetch_item_by_url(url: str) -> dict | None:
    """Return the item matching *url*, or None if not yet saved."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, url, title, summary, content, tags_json, status, created_at "
            "FROM items WHERE url=?",
            (url,),
        ) as cur:
            row = await cur.fetchone()
    return _row_to_dict(row) if row else None


async def fetch_item_with_embedding(id: str) -> dict | None:
    """Return a single item including raw embedding_data blob."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, url, title, summary, content, tags_json, embedding_data, status, created_at "
            "FROM items WHERE id=?",
            (id,),
        ) as cur:
            row = await cur.fetchone()
    return _row_to_dict(row) if row else None


async def save_item(item: dict) -> None:
    """Insert a new item.

    Expected keys: id, url, title, summary, content, tags_json,
                   embedding_data (bytes), status
    """
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            "INSERT INTO items "
            "(id, url, title, summary, content, tags_json, embedding_data, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item["id"],
                item["url"],
                item["title"],
                item["summary"],
                item["content"],
                item.get("tags_json", "[]"),
                item.get("embedding_data", b""),
                item.get("status", "unread"),
            ),
        )
        await db.commit()


async def delete_item(id: str) -> None:
    """Delete an item by ID."""
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute("DELETE FROM items WHERE id=?", (id,))
        await db.commit()


async def update_item_status(id: str, status: str) -> None:
    """Update the status field of an item."""
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute("UPDATE items SET status=? WHERE id=?", (status, id))
        await db.commit()


async def text_search(query: str) -> list[dict]:
    """LIKE search across title, summary, and tags_json.
    Returns items without embedding_data.
    """
    pattern = f"%{query}%"
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, url, title, summary, tags_json, status, created_at "
            "FROM items "
            "WHERE title LIKE ? OR summary LIKE ? OR tags_json LIKE ? "
            "ORDER BY created_at DESC",
            (pattern, pattern, pattern),
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def save_log(
    url: str,
    source: str,
    status: str,
    title: str | None = None,
    error: str | None = None,
) -> int:
    """Insert a log entry and return its auto-increment ID."""
    async with aiosqlite.connect(_db_path()) as db:
        cur = await db.execute(
            "INSERT INTO ingest_log (url, source, status, title, error_message) "
            "VALUES (?, ?, ?, ?, ?)",
            (url, source, status, title, error),
        )
        await db.commit()
        return cur.lastrowid  # type: ignore[return-value]


async def delete_log_entry(entry_id: int) -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute("DELETE FROM ingest_log WHERE id=?", (entry_id,))
        await db.commit()


async def clear_log() -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute("DELETE FROM ingest_log")
        await db.commit()


async def fetch_log(limit: int = 200) -> list[dict]:
    """Return the most recent log entries."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, timestamp, url, source, status, title, error_message "
            "FROM ingest_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def fetch_item_weeks() -> list[dict]:
    """Return one row per week that has items, newest first."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT
                date(created_at, '-' || ((strftime('%w', created_at) + 6) % 7) || ' days') AS week_start,
                COUNT(*) AS item_count
            FROM items
            GROUP BY week_start
            ORDER BY week_start DESC
            """
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def fetch_items_for_week(week_start: str) -> list[dict]:
    """Return items whose week (Mon–Sun) starts on week_start (YYYY-MM-DD)."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, url, title, summary, tags_json, created_at FROM items "
            "WHERE date(created_at, '-' || ((strftime('%w', created_at) + 6) % 7) || ' days') = ? "
            "ORDER BY created_at ASC",
            (week_start,),
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def save_digest(week_start: str, week_end: str, summary: str, item_count: int) -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            "INSERT INTO weekly_digests (week_start, week_end, summary, item_count) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(week_start) DO UPDATE SET "
            "week_end=excluded.week_end, summary=excluded.summary, "
            "item_count=excluded.item_count, created_at=CURRENT_TIMESTAMP",
            (week_start, week_end, summary, item_count),
        )
        await db.commit()


async def fetch_digest(week_start: str) -> dict | None:
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM weekly_digests WHERE week_start=?", (week_start,)
        ) as cur:
            row = await cur.fetchone()
    return _row_to_dict(row) if row else None


async def fetch_all_digests() -> list[dict]:
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM weekly_digests ORDER BY week_start DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def fetch_all_items_with_content() -> list[dict]:
    """Return all items including content, ordered newest first."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, url, title, summary, content, tags_json, status, created_at "
            "FROM items ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def update_item_embedding(id: str, embedding_data: bytes) -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute("UPDATE items SET embedding_data=? WHERE id=?", (embedding_data, id))
        await db.commit()


async def fetch_items_with_embeddings() -> list[dict]:
    """Return all items that have a non-empty embedding_data blob.

    Returned fields: id, url, title, tags_json, embedding_data, created_at.
    """
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, url, title, tags_json, embedding_data, created_at "
            "FROM items WHERE embedding_data IS NOT NULL AND length(embedding_data) > 0 "
            "ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]
