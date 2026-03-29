---
title: VenD RAG Talent Matching
emoji: 🎯
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

## Docker Run

Build and start the app:

```bash
docker compose build
docker compose up -d
```

Open the app:

```text
http://localhost:8501
```

Check logs:

```bash
docker compose logs -f app
```

Stop the app:

```bash
docker compose down
```

Optional ingestion run (requires valid API keys in .env):

```bash
docker compose run --rm app python -m pipeline.ingest
```
