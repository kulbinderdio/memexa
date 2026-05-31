# Memexa

<p align="center">
  <img src="static/icon-256.png" alt="Memexa" width="120">
</p>

> **Your knowledge, connected.**

---
<img width="1384" height="927" alt="image" src="https://github.com/user-attachments/assets/82e73519-9345-4e5a-b7e0-77665287bcf1" />


## The story

I have a bad habit: I skim dozens of articles on my phone every day — on the bus, between meetings, at 11pm when I should be sleeping — and I tell myself I'll read them properly later. For years my solution was a WhatsApp group with just myself in it. I'd paste links in, they'd pile up, and I'd almost never go back to them.

But I still *wanted* them. Some corner of my brain kept insisting these things were worth knowing. I wasn't ready to let them go, I just didn't have the time or the right environment to properly engage with them when I first found them.

So I built Memexa.

The idea is simple: send a URL to a Telegram bot, and Memexa takes care of everything else. It fetches the article, extracts the text, generates an AI summary, tags it, creates a semantic embedding, and files it in your personal knowledge base — all automatically, while you get on with your life. When you do have a moment to explore, your library is waiting: searchable, mapped by topic, and ready to be synthesised into answers.

No more link graveyards. No more guilt about the things you meant to read.

---

## Features

- **Telegram ingestion** — send any URL to your bot and it appears in your library within seconds
- **Manual & PDF ingestion** — paste URLs or upload PDFs directly from the web UI
- **AI summaries & tags** — every item is automatically summarised and tagged using a local or cloud LLM
- **Semantic search** — find items by meaning, not just keywords, using vector embeddings
- **Knowledge map** — a 2D visual map of your library where semantically similar items cluster together
- **AI synthesis** — ask a question across your entire library and get a grounded answer with source citations
- **Weekly digest** — generate an on-demand AI narrative of each week's reading, with links back to every source article
- **Activity feed** — see every ingestion attempt, retry failures, and clear the log
- **Smart extraction** — trafilatura strips ads, navigation, and boilerplate to extract clean article text; Playwright headless Chromium handles JavaScript-rendered pages; archive.ph is an optional fallback for paywalled content
- **Local-first** — runs entirely on your machine; your data never leaves unless you choose a cloud LLM
- **Multi-provider LLM** — works with Ollama (local), OpenAI, or Claude

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (UI)                             │
│            Vanilla JS SPA  ·  Canvas map  ·  SSE               │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────────────────┐
│                    FastAPI Server                               │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │  Telegram   │   │  Ingestion   │   │   REST API          │  │
│  │  Poller     │──▶│  Queue       │   │  /api/items         │  │
│  └─────────────┘   │  (asyncio)   │   │  /api/search        │  │
│                    └──────┬───────┘   │  /api/synthesise     │  │
│                           │           │  /api/map            │  │
│                    ┌──────▼───────┐   │  /api/feed          │  │
│                    │  Extractor   │   └─────────────────────┘  │
│                    │  httpx       │                             │
│                    │  + Playwright│                             │
│                    │  (fallback)  │                             │
│                    └──────┬───────┘                             │
│                           │                                     │
│                    ┌──────▼───────┐                             │
│                    │  LLM Layer   │                             │
│                    │  embed()     │                             │
│                    │  summarise() │                             │
│                    │  chat()      │                             │
│                    └──────┬───────┘                             │
│                           │                                     │
│                    ┌──────▼───────┐                             │
│                    │   SQLite     │                             │
│                    │   (aiosqlite)│                             │
│                    └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┴─────────────────┐
          │                                  │
   ┌──────▼──────┐                   ┌───────▼──────┐
   │   Ollama    │                   │  OpenAI /    │
   │  (local)    │                   │  Claude API  │
   └─────────────┘                   └──────────────┘
```

### Ingestion pipeline

Every URL passes through the same pipeline:

1. **Duplicate check** — look up the URL in SQLite; skip if already saved
2. **Extraction** — `httpx` fetches the page with realistic browser headers and **trafilatura** extracts clean article text, stripping ads, navigation, and boilerplate automatically. If fewer than 500 characters are recovered (JavaScript-rendered pages, SPAs), a headless **Playwright** Chromium instance fetches and renders the page fully before re-running trafilatura. If archive fallback is enabled in Settings, a final attempt is made against the most recent **archive.ph** snapshot.
3. **Embedding** — the title + first 1 200 characters are sent to the embed model (default: `mxbai-embed-large` via Ollama). The resulting float vector is packed into a compact binary blob and stored in SQLite.
4. **Summarisation** — the full article text is sent to the chat model (default: `gemma4:e4b` via Ollama) with a prompt that asks for a 2–4 sentence summary and 3–7 keyword tags, returned as JSON.
5. **Save** — the item (title, summary, content, tags, embedding) is written to SQLite and broadcast to all connected browsers over **Server-Sent Events**.

### Database schema

```sql
-- AI-generated weekly digest summaries
CREATE TABLE weekly_digests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start  DATE NOT NULL UNIQUE,  -- Monday of the week (YYYY-MM-DD)
    week_end    DATE NOT NULL,
    summary     TEXT,                  -- LLM-generated narrative
    item_count  INTEGER,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Saved knowledge items
CREATE TABLE items (
    id             TEXT PRIMARY KEY,
    url            TEXT UNIQUE NOT NULL,
    title          TEXT,
    summary        TEXT,          -- AI-generated summary
    content        TEXT,          -- full extracted article text
    tags_json      TEXT,          -- JSON array of keyword tags
    embedding_data BLOB,          -- packed float32 vector (binary)
    status         TEXT,          -- 'unread' | 'read'
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Every ingestion attempt (success, failure, skipped)
CREATE TABLE ingest_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
    url           TEXT,
    source        TEXT,           -- 'telegram' | 'manual' | 'upload' | 'retry'
    status        TEXT,           -- 'success' | 'failed' | 'skipped'
    title         TEXT,
    error_message TEXT
);

-- Key/value settings (LLM provider, model names, Telegram token, etc.)
CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
```

### Semantic search & knowledge map

Every saved item has a high-dimensional embedding vector stored as a binary blob in SQLite. At search time, the query is embedded with the same model and cosine similarity is computed in-process with NumPy — no vector database required.

The **knowledge map** projects all embeddings to 2D using a from-scratch PCA implementation (power iteration, no scikit-learn dependency) and renders them on an HTML5 Canvas. Items that are semantically related cluster together; clicking a dot opens the item in the library.

### AI synthesis

The synthesis endpoint selects the top 5 most semantically relevant items for a given question (via cosine similarity), assembles them as context, and sends a grounded prompt to the chat model. The response cites source numbers so you can trace every claim back to the original article.

### Real-time updates

The server maintains an SSE (Server-Sent Events) endpoint at `/api/events`. The browser holds a persistent connection; as each ingestion stage completes, events (`ingestion_started`, `item_added`, `ingestion_failed`) are pushed to all connected clients without polling.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn |
| Database | SQLite via aiosqlite |
| Web scraping | httpx, trafilatura |
| JS rendering | Playwright (Chromium headless) |
| PDF extraction | pypdf |
| LLM (local) | Ollama (`mxbai-embed-large` + `gemma4:e4b`) |
| LLM (cloud) | OpenAI API, Anthropic Claude API |
| Embeddings | NumPy (cosine similarity, PCA) |
| Messaging | Telegram Bot API |
| Frontend | Vanilla JS, HTML5 Canvas |

---

## Getting started

### Docker (recommended)

The easiest way to run Memexa. Ollama, all models, and the app start together.

#### 1. Clone and start

```bash
git clone https://github.com/yourusername/memexa-web.git
cd memexa-web

# Without GPU (CPU only)
docker compose up -d

# With Nvidia GPU (recommended)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

The first run downloads `mxbai-embed-large` (~670 MB) and `gemma4:e4b` automatically in the background. A banner in the UI shows live download progress and dismisses itself once both models are ready.

Watch server logs if you prefer the terminal:

```bash
docker compose logs -f memexa
```

Open **http://localhost:7700** straight away — the banner will keep you informed while models download.

#### 2. Set up Telegram (optional but recommended)

1. Message [@BotFather](https://t.me/botfather) on Telegram → `/newbot` → follow the prompts → copy the token
2. Open **http://localhost:7700** → Settings (gear icon) → paste the token → Save
3. Send any URL to your bot — it appears in your library within seconds

#### 3. Persistent storage

All data survives container restarts and upgrades via named Docker volumes:

| Volume | Contents |
|---|---|
| `memexa-data` | SQLite database, uploaded PDFs |
| `ollama-models` | Downloaded Ollama models (~4 GB) |

#### 4. Common commands

> If you're using GPU, replace `docker compose` with `docker compose -f docker-compose.yml -f docker-compose.gpu.yml` in all commands below.

```bash
# Start everything (CPU)
docker compose up -d

# Start everything (Nvidia GPU)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Stop everything (data is preserved)
docker compose down

# View live logs
docker compose logs -f memexa

# Restart just the app (after a config change)
docker compose restart memexa

# Open a shell inside the app container
docker compose exec memexa bash

# Pull a different Ollama model
docker compose exec ollama ollama pull gemma4:e4b
```

#### 5. Updating Memexa

```bash
git pull

# CPU
docker compose build memexa && docker compose up -d memexa

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build memexa
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d memexa
```

Your data volumes are untouched by updates.

#### 6. Backup

The entire database and uploads live in the `memexa-data` volume. To back it up:

```bash
docker run --rm \
  -v memexa-web_memexa-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/memexa-backup.tar.gz -C /data .
```

Restore:

```bash
docker run --rm \
  -v memexa-web_memexa-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/memexa-backup.tar.gz -C /data
```

#### 7. GPU acceleration (Nvidia)

A `docker-compose.gpu.yml` override is included. Start with it layered on top of the base compose file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

This passes all Nvidia GPUs through to the Ollama container. Make sure the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed on the host first.

#### 8. Troubleshooting

**App says "server may be unavailable"**
Check that Ollama is healthy: `docker compose ps` — the `ollama` service should show `healthy`. If it shows `starting`, wait 20–30 seconds and refresh.

**Models not found / embedding errors**
The app pulls missing models automatically on startup. Check the banner in the UI or run:
```bash
docker compose logs -f memexa
```

**Port 7700 already in use**
Change the host port in `docker-compose.yml`:
```yaml
ports:
  - "8080:7700"   # access on http://localhost:8080
```

**Want to use OpenAI or Claude instead of Ollama**
Open Settings in the UI, switch the provider, and enter your API key. The Ollama container will sit idle but won't cause errors.

---

### Manual installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running (for local LLM)
- A Telegram bot token (from [@BotFather](https://t.me/botfather)) — optional but recommended

### Installation

```bash
git clone https://github.com/yourusername/memexa-web.git
cd memexa-web

pip install -r requirements.txt
playwright install chromium
```

### Pull the default Ollama models

```bash
ollama pull mxbai-embed-large   # embeddings
ollama pull gemma4:e4b             # chat / summarisation
```

### Run

```bash
python server.py
# Open http://localhost:7700
```

### Configure

Open **Settings** in the sidebar to set:

- **LLM provider** — Ollama (default), OpenAI, or Claude
- **Ollama models** — chat and embedding model names
- **Telegram bot token** — paste your token; the poller starts immediately

### Set up the Telegram bot

1. Message [@BotFather](https://t.me/botfather) → `/newbot`
2. Copy the token into Memexa Settings
3. Send any URL to your bot — it appears in your library within a few seconds

---

## Project structure

```
memexa-web/
├── server.py          # FastAPI app, ingestion pipeline, API routes
├── extractor.py       # httpx + BeautifulSoup + Playwright extraction
├── llm.py             # LLM provider abstraction (Ollama / OpenAI / Claude)
├── db.py              # SQLite schema and async data access layer
├── pca.py             # Pure-NumPy PCA for the knowledge map
├── telegram_poller.py # Long-polling Telegram bot
├── requirements.txt
└── static/
    └── index.html     # Single-file SPA (HTML + CSS + JS)
```

---

## License

MIT
