#!/bin/bash
# =============================================================================
# PropsAI Backend Deploy Script for Hostinger VPS (Docker + Traefik)
# Run this once to set up, or again to update to latest code.
# =============================================================================
set -e

APP_DIR="/opt/propsai"
REPO="https://github.com/cjg131/propsai.git"

echo "============================================"
echo "  PropsAI Backend Deploy"
echo "============================================"

# ── Step 1: Clone or pull the repo ──
if [ -d "$APP_DIR" ]; then
    echo "[1/5] Updating existing repo..."
    cd "$APP_DIR"
    git pull origin main
else
    echo "[1/5] Cloning repo..."
    git clone "$REPO" "$APP_DIR"
    cd "$APP_DIR"
fi

cd "$APP_DIR/backend"

# ── Step 2: Check for .env file ──
if [ ! -f .env ]; then
    echo ""
    echo "[2/5] No .env file found. Creating from template..."
    cp .env.example .env
    echo ""
    echo "  ⚠️  IMPORTANT: Edit your .env file with your API keys!"
    echo "  Run: nano $APP_DIR/backend/.env"
    echo ""
    echo "  At minimum you need:"
    echo "    - SUPABASE_URL and SUPABASE_KEY"
    echo "    - SPORTSDATAIO_API_KEY"
    echo "    - OPENAI_API_KEY"
    echo "    - KALSHI_API_KEY_ID"
    echo "    - PAPER_MODE=true  (already set)"
    echo ""
    echo "  Also copy your kalshi.key file to: $APP_DIR/backend/kalshi.key"
    echo ""
    echo "  After editing .env and adding kalshi.key, run this script again."
    exit 0
else
    echo "[2/5] .env file found ✓"
fi

# ── Step 3: Check for Kalshi key ──
if [ ! -f kalshi.key ]; then
    echo ""
    echo "  ⚠️  kalshi.key not found!"
    echo "  Copy your Kalshi RSA private key to: $APP_DIR/backend/kalshi.key"
    echo "  Then run this script again."
    exit 0
else
    echo "[3/5] kalshi.key found ✓"
fi

# ── Step 4: Build and start ──
echo "[4/5] Building Docker image (this takes 2-3 minutes first time)..."
docker compose build --no-cache

echo "[5/5] Starting PropsAI backend..."
docker compose up -d

echo ""
echo "============================================"
echo "  ✓ PropsAI is running!"
echo "============================================"
echo ""
echo "  Container status:"
docker compose ps
echo ""
echo "  View logs:        docker compose logs -f"
echo "  Stop:             docker compose down"
echo "  Update & redeploy: cd $APP_DIR && git pull && cd backend && docker compose up -d --build"
echo ""
echo "  Health check:     curl http://localhost:8000/health"
echo "  Agent status:     curl http://localhost:8000/agent/status"
echo "  Learning status:  curl http://localhost:8000/agent/learning/status"
echo ""
echo "  Paper mode is ON by default. To go live:"
echo "  1. Edit $APP_DIR/backend/.env"
echo "  2. Set PAPER_MODE=false"
echo "  3. Run: docker compose restart"
echo ""
