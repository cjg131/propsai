# PropsAI Backend

Python/FastAPI backend for the PropsAI NBA player props prediction engine.

## Setup

```bash
poetry install
cp .env.example .env
# Fill in environment variables
poetry run uvicorn app.main:app --reload
```

Readiness and live start:

```bash
python3 scripts/live_readiness_check.py
curl http://localhost:8000/api/kalshi/agent/readiness
curl -X POST http://localhost:8000/api/kalshi/agent/start-live-weather
```

Generic `POST /api/kalshi/agent/start` is paper-only. Live startup is weather-only by design.
