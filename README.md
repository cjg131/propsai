# PropsAI

NBA player props prediction platform powered by ensemble ML models with a full-featured dashboard for picks, research, odds comparison, bet tracking, performance analytics, and backtesting.

## Architecture

- **Frontend**: Next.js 16 + TailwindCSS v4 + shadcn/ui + Recharts + TanStack Query
- **Backend**: Python 3.9+ + FastAPI + Poetry
- **Database**: Supabase (PostgreSQL) — 15+ tables
- **ML Models**: XGBoost, Random Forest, Logistic Regression, LSTM, Transformer, Bayesian (sklearn BayesianRidge)
- **Deploy**: Netlify (frontend) + Railway (backend)

## Features

| Page | Description |
|------|-------------|
| **Today's Picks** | Filterable prediction table with expandable model transparency rows |
| **Player Research** | Search, season stats, rolling averages, AI scouting reports (GPT-4) |
| **Odds Comparison** | Side-by-side sportsbook odds with best-line highlighting |
| **Bet Tracker** | Summary cards, filters, bet history with P&L tracking |
| **Parlay Builder** | AI-suggested correlated parlays + manual builder with slip |
| **Bankroll** | Equity curve chart, balance tracking, high water mark |
| **Performance** | Cumulative P&L chart, prop type breakdown, drawdown analysis |
| **Backtesting** | Historical simulation with config panel and progress tracking |
| **Paper Trading** | Virtual bankroll with auto-logged predictions |
| **Data Management** | Refresh data, retrain models, export, API quota monitoring |
| **Settings** | Bankroll config, model presets (Conservative/Balanced/Aggressive), API keys |

## Project Structure

```
/frontend              → Next.js app (11 pages, React Query hooks, shadcn/ui)
  /src/app/            → Page routes (/, /players, /odds, /bets, /parlays, etc.)
  /src/lib/hooks/      → TanStack Query hooks (predictions, bets, players, data, settings)
  /src/lib/api.ts      → API client
  /src/components/     → Shared components (sidebar, theme toggle)
/backend               → Python/FastAPI (7 API modules, 6 ML models)
  /app/api/            → FastAPI routes (predictions, players, bets, odds, data, settings)
  /app/models/         → ML models (xgboost, random_forest, logistic, lstm, transformer, bayesian, ensemble)
  /app/services/       → SportsDataIO client, Supabase client, injury scraper, data manager
  /app/schemas/        → Pydantic response/request models
  /app/utils/          → Kelly criterion, travel distance, fantasy scoring
  /scripts/            → Database schema SQL, seed script
  /tests/              → 24 unit tests
.github/workflows      → CI/CD pipeline
```

## Getting Started

### Prerequisites

- Node.js 20 LTS
- Python 3.9+
- Poetry (`pip install poetry`)
- Supabase account (free tier)
- SportsDataIO API key

### 1. Database Setup

Create a Supabase project, then run the schema in the SQL Editor:

```bash
# Copy the contents of backend/scripts/schema.sql into Supabase SQL Editor and run it.
# This creates all 15+ tables, indexes, triggers, and seeds built-in model presets.
```

### 2. Backend Setup

```bash
cd backend
poetry install
cp .env.example .env
# Fill in: SUPABASE_URL, SUPABASE_KEY, SPORTSDATAIO_API_KEY, OPENAI_API_KEY
poetry run uvicorn app.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health` → `{"status": "healthy"}`

### 3. Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open: `http://localhost:3000`

### 4. Load Data (Optional)

```bash
cd backend
poetry run python scripts/seed_historical.py --seasons 3
```

Or use the Data Management page in the UI to refresh today's data and retrain models.

## Running Tests

```bash
# Backend
cd backend
poetry run pytest tests/ -v

# Frontend
cd frontend
npm run build   # Type-checks + builds all 11 routes
npx tsc --noEmit  # TypeScript only
```

## Environment Variables

See `.env.example` files in `/frontend` and `/backend` for all required variables.

### Backend (`backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase service role key |
| `SPORTSDATAIO_API_KEY` | Yes | SportsDataIO API key |
| `OPENAI_API_KEY` | Yes | OpenAI API key (for scouting reports) |
| `TWITTER_BEARER_TOKEN` | No | Twitter/X API for injury news |
| `SENTRY_DSN` | No | Sentry error tracking |

### Frontend (`frontend/.env.local`)

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes | Backend URL (default: `http://localhost:8000`) |

## Deployment

### Frontend → Netlify

```bash
cd frontend
# Build command: npm run build
# Publish directory: .next
# Environment: set NEXT_PUBLIC_API_URL to your Railway backend URL
```

### Backend → Railway

```bash
cd backend
# Start command: poetry run uvicorn app.main:app --host 0.0.0.0 --port $PORT
# Environment: set all backend env vars in Railway dashboard
```

## External Services

| Service | Purpose | Cost |
|---------|---------|------|
| Supabase | Database (PostgreSQL) | Free tier |
| SportsDataIO | NBA stats, odds, injuries | ~$25/month |
| OpenAI | GPT-4 scouting reports | ~$0.01-0.03/report |
| Twitter/X API | Injury news scraping | Optional |
| Sentry | Error tracking | Free tier |
| Netlify | Frontend hosting | Free tier |
| Railway | Backend hosting | Free tier |

## License

Private - All rights reserved.
