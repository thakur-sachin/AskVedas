# Scripture RAG Backend

FastAPI backend for a Scripture RAG mobile app with 3 modes: Ask, Daily Wisdom, Mood Guidance.

- Vector DB: Qdrant (`scriptures_en`, `scriptures_hi`)
- PDFs: exactly 7 local files under `data/`
- OCR: Hindi/Sanskrit scans via Tesseract (`hin+san`, fallback if `san` missing)
- LLM: Ollama provider with deterministic fallback
- Citation safety: no retrieved citation => no answer

## Project Structure

```text
repo/
  app/
    main.py
    api/router.py
    api/v1/{health.py,ready.py,query.py,daily.py,library.py,feedback.py,admin.py}
    core/{config.py,logging.py,errors.py}
    models/{schemas.py,domain.py}
    services/{language.py,retriever.py,generator.py,daily_service.py,library_service.py,feedback_service.py,admin_service.py,embeddings.py}
    vector/{qdrant_client.py,collections.py,upsert.py}
    ingest/{ingest_en.py,ingest_hi_ocr.py,pdf_extract.py,ocr.py,chunking.py,cleaning.py}
    db/{session.py,models.py}
  scripts/{run_local_qdrant.sh,ingest_all.sh,dev_run.sh}
  docker-compose.yml
  Dockerfile
  pyproject.toml
  .env.example
  README.md
```

## Setup

```bash
cp .env.example .env
# update ADMIN_TOKEN in .env
```

## Run with Docker Compose

```bash
docker compose up -d --build
```

API: `http://localhost:8000`
Qdrant: `http://localhost:6333`

### Use Qdrant Cloud (optional)

Set these in `.env` and restart API:

```bash
QDRANT_URL=https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-api-key>
```

When `QDRANT_URL` is set, the app uses cloud Qdrant; local `QDRANT_HOST/QDRANT_PORT` are ignored.

## Ingest All 7 Docs

```bash
./scripts/ingest_all.sh
```

Check job status:

```bash
curl -s "http://localhost:8000/api/v1/admin/jobs/<JOB_ID>" -H "X-Admin-Token: change-me"
```

## Example curl

Health:

```bash
curl -s http://localhost:8000/api/v1/health
```

Query:

```bash
curl -s http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "mode":"ask",
    "query":"What do the Upanishads say about self-knowledge?",
    "mood":null,
    "lang_pref":"auto",
    "tone":"neutral",
    "sources":{"upanishads":true,"puranas":false,"vedas":true},
    "k":6,
    "include_original":true,
    "session_id":"demo-1"
  }'
```

Daily:

```bash
curl -s "http://localhost:8000/api/v1/daily?lang=auto"
```

Refresh daily (admin):

```bash
curl -s -X POST "http://localhost:8000/api/v1/daily/refresh?lang=auto" \
  -H "X-Admin-Token: change-me"
```

## Local Dev (without Docker)

```bash
pip install -e .[test]
./scripts/run_local_qdrant.sh
./scripts/dev_run.sh
```

## Notes

- If retrieval score is weak or no chunk is found, API returns:
  `I don’t find a direct passage in the indexed scriptures for this question.`
- Citations are always sourced from Qdrant payload (`doc`, `page`, `chunk_id`), never fabricated.
- `POST /admin/ingest` runs true background ingestion jobs and persists status in SQLite (`jobs` table).
