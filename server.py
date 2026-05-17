"""
memexa-web — FastAPI server.

Entry point: python server.py [--host 0.0.0.0] [--port 7700]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import db as database
from db import (
    clear_log,
    delete_item,
    delete_log_entry,
    fetch_all_items,
    fetch_item,
    fetch_item_by_url,
    fetch_items_with_embeddings,
    fetch_log,
    init_db,
    pack_embedding,
    save_item,
    save_log,
    text_search,
    unpack_embedding,
    update_item_status,
    update_setting,
)
from extractor import extract
from llm import get_provider
from pca import pca_2d
from telegram_poller import TelegramPoller

# ---------------------------------------------------------------------------
# Static directory
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

_UPLOADS_DIR = Path.home() / ".memexa-web" / "uploads"

import datetime as _dt


def _to_unix(ts: str | None) -> float:
    """Convert a SQLite DATETIME string to a Unix timestamp (float seconds).
    Returns 0 if the value is missing or unparseable."""
    if not ts:
        return 0.0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return _dt.datetime.strptime(ts, fmt).replace(
                tzinfo=_dt.timezone.utc
            ).timestamp()
        except ValueError:
            continue
    return 0.0


def serialize_item(row: dict, extra: dict | None = None) -> dict:
    """Normalize a DB row dict to the shape the frontend expects."""
    tags_raw = row.get("tags_json", "[]")
    try:
        tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
    except Exception:
        tags = []
    out = {
        "id": row["id"],
        "url": row["url"],
        "title": row.get("title", ""),
        "summary": row.get("summary", ""),
        "tags": tags,
        "status": row.get("status", "unread"),
        "createdAt": _to_unix(row.get("created_at")),
    }
    if row.get("content"):
        out["content"] = row["content"]
    if extra:
        out.update(extra)
    return out

# ---------------------------------------------------------------------------
# SSE Event Bus
# ---------------------------------------------------------------------------


class EventBus:
    """Simple fan-out SSE publisher."""

    def __init__(self) -> None:
        self.queues: list[asyncio.Queue] = []

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self.queues.remove(q)
        except ValueError:
            pass

    async def publish(self, data: dict) -> None:
        payload = json.dumps(data)
        dead: list[asyncio.Queue] = []
        for q in list(self.queues):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)


event_bus = EventBus()

# ---------------------------------------------------------------------------
# Ingestion queue & worker
# ---------------------------------------------------------------------------

_ingest_queue: asyncio.Queue = asyncio.Queue()


async def process_url(url: str, source: str = "manual") -> None:
    """Full ingestion pipeline for a single URL."""
    # 1. Duplicate check — skip silently for retries so the old failed entry
    #    doesn't get joined by a spurious "Duplicate URL" log entry.
    if await fetch_item_by_url(url):
        if source != "retry":
            await save_log(url, source, "skipped", error="Duplicate URL")
        return

    await event_bus.publish({"type": "ingestion_started", "url": url})

    try:
        # 2. Extract content
        extracted = await extract(url)
        title = extracted.title
        text = extracted.text

        # 3. Minimum content check
        if len(text) < 150:
            raise ValueError(f"Insufficient content ({len(text)} chars)")

        # 4. Embed title + truncated body
        provider = await get_provider()
        embed_input = f"{title}\n\n{text[:1200]}"
        embedding = await provider.embed(embed_input)
        embedding_bytes = pack_embedding(embedding)

        # 5. Summarise
        summary, tags = await provider.summarise(text)

        # 6. Save item
        item_id = str(uuid.uuid4())
        tags_json = json.dumps(tags)
        item: dict = {
            "id": item_id,
            "url": url,
            "title": title,
            "summary": summary,
            "content": text,
            "tags_json": tags_json,
            "embedding_data": embedding_bytes,
            "status": "unread",
        }
        await save_item(item)

        # 7. Log success
        await save_log(url, source, "success", title=title)

        # 8. Broadcast
        public_item = serialize_item({
            "id": item_id,
            "url": url,
            "title": title,
            "summary": summary,
            "tags_json": tags_json,
            "status": "unread",
            "created_at": "",
        })
        await event_bus.publish({"type": "item_added", "item": public_item})

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        if "UNIQUE constraint failed" in error_msg:
            # Race condition — another task inserted this URL first
            await save_log(url, source, "skipped", error="Already saved")
        else:
            await save_log(url, source, "failed", error=error_msg)
            await event_bus.publish(
                {"type": "ingestion_failed", "url": url, "error": error_msg}
            )


async def process_pdf_item(url: str, source: str, pdf_path: Path) -> None:
    """Ingestion pipeline for an uploaded PDF file."""
    await event_bus.publish({"type": "ingestion_started", "url": url})

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)

        # Title: PDF metadata → filename stem
        meta_title = ""
        if reader.metadata:
            meta_title = (reader.metadata.get("/Title") or "").strip()
        if not meta_title:
            # url format is pdf://{item_id}/{filename}
            filename = url.split("/", 2)[-1] if "/" in url[6:] else url[6:]
            meta_title = Path(filename).stem.replace("_", " ").replace("-", " ")

        title = meta_title or "Untitled PDF"

        # Extract text from all pages
        pages: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    pages.append(t)
            except Exception:
                continue

        text = "\n\n".join(pages)

        if len(text) < 50:
            raise ValueError(
                f"Could not extract text from this PDF ({page_count} pages). "
                "It may be scanned or image-only."
            )

        provider = await get_provider()
        embed_input = f"{title}\n\n{text[:1200]}"
        embedding = await provider.embed(embed_input)
        embedding_bytes = pack_embedding(embedding)

        summary, tags = await provider.summarise(text)

        item_id = url.split("://")[1].split("/")[0]  # extract from pdf://{item_id}/...
        tags_json = json.dumps(tags)
        item: dict = {
            "id": item_id,
            "url": url,
            "title": title,
            "summary": summary,
            "content": text,
            "tags_json": tags_json,
            "embedding_data": embedding_bytes,
            "status": "unread",
        }
        await save_item(item)
        await save_log(url, source, "success", title=title)

        public_item = serialize_item({
            "id": item_id,
            "url": url,
            "title": title,
            "summary": summary,
            "tags_json": tags_json,
            "status": "unread",
            "created_at": "",
        })
        await event_bus.publish({"type": "item_added", "item": public_item})

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        if "UNIQUE constraint failed" in error_msg:
            await save_log(url, source, "skipped", error="Already saved")
        else:
            await save_log(url, source, "failed", error=error_msg)
            await event_bus.publish(
                {"type": "ingestion_failed", "url": url, "error": error_msg}
            )


async def ingestion_worker() -> None:
    """Consume items from the queue one at a time."""
    while True:
        task: dict = await _ingest_queue.get()
        url = task["url"]
        source = task["source"]
        pdf_path = task.get("pdf_path")
        try:
            if pdf_path:
                await process_pdf_item(url, source, Path(pdf_path))
            else:
                await process_url(url, source)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] Unhandled error for {url}: {exc}", file=sys.stderr)
        finally:
            _ingest_queue.task_done()


# ---------------------------------------------------------------------------
# Telegram poller
# ---------------------------------------------------------------------------

async def _on_telegram_url(url: str) -> None:
    await _ingest_queue.put({"url": url, "source": "telegram"})


telegram_poller = TelegramPoller(on_url_found=_on_telegram_url)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    asyncio.create_task(ingestion_worker(), name="ingestion-worker")

    settings = await database.get_settings()
    token = settings.get("telegram_bot_token", "").strip()
    if token:
        await telegram_poller.start(token)

    yield

    await telegram_poller.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="memexa-web", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Key masking helper
# ---------------------------------------------------------------------------

def _mask_key(value: str) -> str:
    """Show only last 4 chars of an API key; blank if short."""
    if not value or len(value) < 5:
        return ""
    return "****" + value[-4:]


_SENSITIVE_KEYS = {"openai_api_key", "claude_api_key", "telegram_bot_token"}


def _mask_settings(settings: dict) -> dict:
    masked = dict(settings)
    for k in _SENSITIVE_KEYS:
        if k in masked and masked[k]:
            masked[k] = _mask_key(masked[k])
    return masked


# ---------------------------------------------------------------------------
# Routes: static files
# ---------------------------------------------------------------------------

# Serve index.html at /
@app.get("/", include_in_schema=False)
async def root() -> Response:
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return Response(content=index.read_bytes(), media_type="text/html")
    return Response(content="<h1>memexa-web</h1><p>No index.html found in static/</p>", media_type="text/html")


# Mount /static for assets
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes: PDF upload & serve
# ---------------------------------------------------------------------------

@app.post("/api/upload", status_code=202)
async def upload_pdf(file: UploadFile = File(...)) -> JSONResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    item_id = str(uuid.uuid4())
    safe_name = Path(file.filename or "document.pdf").name
    pdf_path = _UPLOADS_DIR / f"{item_id}.pdf"
    pdf_path.write_bytes(await file.read())

    url = f"pdf://{item_id}/{safe_name}"
    await _ingest_queue.put({"url": url, "source": "upload", "pdf_path": str(pdf_path)})
    return JSONResponse({"status": "queued", "filename": safe_name})


@app.get("/api/uploads/{item_id}")
async def serve_pdf(item_id: str) -> FileResponse:
    pdf_path = _UPLOADS_DIR / f"{item_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(str(pdf_path), media_type="application/pdf")


# ---------------------------------------------------------------------------
# Routes: items
# ---------------------------------------------------------------------------

@app.get("/api/items")
async def list_items() -> JSONResponse:
    items = await fetch_all_items()
    return JSONResponse([serialize_item(r) for r in items])


@app.get("/api/items/{item_id}")
async def get_item(item_id: str) -> JSONResponse:
    item = await fetch_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    await update_item_status(item_id, "read")
    return JSONResponse(serialize_item(item, {"status": "read"}))


@app.post("/api/ingest", status_code=202)
async def ingest_url(body: dict) -> JSONResponse:
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    await _ingest_queue.put({"url": url, "source": "manual"})
    return JSONResponse({"status": "queued", "url": url})


@app.delete("/api/items/{item_id}", status_code=204)
async def remove_item(item_id: str) -> Response:
    item = await fetch_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    await delete_item(item_id)
    return Response(status_code=204)


@app.post("/api/items/{item_id}/read")
async def mark_read(item_id: str) -> JSONResponse:
    item = await fetch_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    await update_item_status(item_id, "read")
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Routes: search
# ---------------------------------------------------------------------------

@app.get("/api/search")
async def keyword_search(q: str = "") -> JSONResponse:
    if not q:
        raise HTTPException(status_code=400, detail="q is required")
    results = await text_search(q)
    return JSONResponse([serialize_item(r) for r in results])


@app.post("/api/search")
async def semantic_search(body: dict) -> JSONResponse:
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    provider = await get_provider()
    try:
        query_vec = await provider.embed(query)
    except NotImplementedError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    rows = await fetch_items_with_embeddings()
    if not rows:
        return JSONResponse([])

    import numpy as np  # local import to keep top-level clean

    q = np.array(query_vec, dtype=np.float64)
    q_norm = np.linalg.norm(q)

    scored: list[tuple[float, dict]] = []
    for row in rows:
        vec = unpack_embedding(row["embedding_data"])
        if not vec:
            continue
        v = np.array(vec, dtype=np.float64)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12 or q_norm < 1e-12:
            score = 0.0
        else:
            score = float(np.dot(q, v) / (q_norm * v_norm))
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, row in scored[:20]:
        results.append(serialize_item(row, {"score": round(score, 4)}))
    return JSONResponse(results)


# ---------------------------------------------------------------------------
# Routes: synthesise
# ---------------------------------------------------------------------------

@app.post("/api/synthesise")
async def synthesise(body: dict) -> JSONResponse:
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    provider = await get_provider()

    # Find top 5 semantically relevant items
    try:
        query_vec = await provider.embed(query)
    except NotImplementedError:
        # Fallback to text search if embeddings not available
        rows = await text_search(query)
        top_rows = rows[:5]
        sources = []
        for r in top_rows:
            full = await fetch_item(r["id"])
            if full:
                sources.append(full)
    else:
        import numpy as np

        q = np.array(query_vec, dtype=np.float64)
        q_norm = np.linalg.norm(q)

        rows = await fetch_items_with_embeddings()
        scored: list[tuple[float, dict]] = []
        for row in rows:
            vec = unpack_embedding(row["embedding_data"])
            if not vec:
                continue
            v = np.array(vec, dtype=np.float64)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12 or q_norm < 1e-12:
                score = 0.0
            else:
                score = float(np.dot(q, v) / (q_norm * v_norm))
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        sources = []
        for _score, row in scored[:5]:
            full = await fetch_item(row["id"])
            if full:
                sources.append(full)

    if not sources:
        return JSONResponse({"answer": "No relevant items found.", "sources": []})

    # Build synthesis prompt
    context_parts = []
    for i, src in enumerate(sources, 1):
        snippet = (src.get("content") or src.get("summary") or "")[:800]
        context_parts.append(
            f"[Source {i}] {src['title']}\nURL: {src['url']}\n{snippet}"
        )
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "user",
            "content": (
                f"Using the following saved articles as context, answer this question:\n"
                f"Question: {query}\n\n"
                f"Context:\n{context}\n\n"
                f"Provide a clear, concise answer citing source numbers where relevant."
            ),
        }
    ]
    answer = await provider.chat(messages)

    source_refs = [
        {"id": s["id"], "url": s["url"], "title": s["title"]} for s in sources
    ]
    return JSONResponse({"answer": answer, "sources": source_refs})


# ---------------------------------------------------------------------------
# Routes: map (PCA)
# ---------------------------------------------------------------------------

@app.get("/api/map")
async def get_map() -> JSONResponse:
    rows = await fetch_items_with_embeddings()
    if not rows:
        return JSONResponse([])

    vectors = [unpack_embedding(r["embedding_data"]) for r in rows]
    # Filter out empty embeddings
    valid: list[tuple[dict, list[float]]] = [
        (r, v) for r, v in zip(rows, vectors) if len(v) > 0
    ]
    if not valid:
        return JSONResponse([])

    valid_rows, valid_vecs = zip(*valid)
    coords = pca_2d(list(valid_vecs))

    result = []
    for row, (x, y) in zip(valid_rows, coords):
        result.append(serialize_item(row, {"x": x, "y": y}))
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Routes: feed / log
# ---------------------------------------------------------------------------

@app.get("/api/feed")
async def get_feed() -> JSONResponse:
    entries = await fetch_log()
    normalized = [
        {
            "id": e["id"],
            "timestamp": _to_unix(e.get("timestamp")),
            "url": e["url"],
            "source": e.get("source", "manual"),
            "status": e["status"],
            "title": e.get("title"),
            "error": e.get("error_message"),
        }
        for e in entries
    ]
    return JSONResponse(normalized)


@app.post("/api/feed/retry")
async def retry_url(body: dict) -> JSONResponse:
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    if await fetch_item_by_url(url):
        return JSONResponse({"status": "already_saved", "url": url})
    entry_id = body.get("entryId")
    if entry_id:
        await delete_log_entry(int(entry_id))
    await _ingest_queue.put({"url": url, "source": "retry"})
    return JSONResponse({"status": "queued", "url": url})


@app.delete("/api/feed/{entry_id}", status_code=204)
async def delete_feed_entry(entry_id: int) -> Response:
    await delete_log_entry(entry_id)
    return Response(status_code=204)


@app.delete("/api/feed", status_code=204)
async def clear_feed() -> Response:
    await clear_log()
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Routes: status
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status() -> JSONResponse:
    """Check whether the configured LLM models are available."""
    settings = await database.get_settings()
    provider = settings.get("llm_provider", "ollama")

    if provider != "ollama":
        return JSONResponse({"ready": True, "provider": provider})

    base_url = os.environ.get("OLLAMA_BASE_URL") or settings.get("ollama_base_url") or "http://localhost:11434"
    chat_model = settings.get("ollama_chat_model") or "gemma3:4b"
    embed_model = settings.get("ollama_embed_model") or "mxbai-embed-large"

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            resp.raise_for_status()
            available = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
    except Exception:
        return JSONResponse({"ready": False, "provider": "ollama", "missing_models": [chat_model, embed_model], "ollama_reachable": False})

    missing = [m for m in [chat_model, embed_model] if m.split(":")[0] not in available]
    return JSONResponse({"ready": not missing, "provider": "ollama", "missing_models": missing, "ollama_reachable": True})


# ---------------------------------------------------------------------------
# Routes: settings
# ---------------------------------------------------------------------------

@app.get("/api/settings")
async def get_settings_route() -> JSONResponse:
    settings = await database.get_settings()
    return JSONResponse(_mask_settings(settings))


@app.put("/api/settings")
async def update_settings_route(body: dict) -> JSONResponse:
    old_settings = await database.get_settings()
    old_token = old_settings.get("telegram_bot_token", "")

    for key, value in body.items():
        if isinstance(value, str):
            await update_setting(key, value)

    new_settings = await database.get_settings()
    new_token = new_settings.get("telegram_bot_token", "").strip()

    # Restart telegram poller if token changed
    if new_token != old_token.strip():
        if new_token:
            await telegram_poller.restart(new_token)
        else:
            await telegram_poller.stop()

    return JSONResponse(_mask_settings(new_settings))


# ---------------------------------------------------------------------------
# Routes: SSE events
# ---------------------------------------------------------------------------

@app.get("/api/events")
async def sse_events() -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        q = await event_bus.subscribe()
        # Send a heartbeat comment immediately so the browser knows the stream is alive
        yield ": connected\n\n"
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive comment every 30 s
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            event_bus.unsubscribe(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="memexa-web server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7700, help="Bind port (default: 7700)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    lan_ip = _get_lan_ip()
    print(f"\n  memexa-web")
    print(f"  Local:   http://localhost:{args.port}")
    print(f"  Network: http://{lan_ip}:{args.port}\n")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
