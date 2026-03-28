FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

WORKDIR /app

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies (no virtualenv inside container)
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root --only main && \
    pip install --no-cache-dir py-clob-client==0.34.6

# Copy app code
COPY . .

# Install the project itself
RUN poetry install --no-interaction --no-ansi --only main

# Create data directory for SQLite + learning state
RUN mkdir -p /app/app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
