# PropsAI Backend

Python/FastAPI backend for the PropsAI NBA player props prediction engine.

## Setup

```bash
poetry install
cp .env.example .env
# Fill in environment variables
poetry run uvicorn app.main:app --reload
```
