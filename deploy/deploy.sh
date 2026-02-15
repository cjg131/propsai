#!/bin/bash
set -euo pipefail

# ─── PropsAI VPS Deploy Script ───────────────────────────────────
# Run from your Mac: ./deploy/deploy.sh <server-ip>
# First-time setup:  ./deploy/deploy.sh <server-ip> --setup

SERVER_IP="${1:?Usage: ./deploy.sh <server-ip> [--setup]}"
SETUP="${2:-}"
SSH_USER="root"
REMOTE_DIR="/opt/propsai"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "🚀 Deploying PropsAI to $SERVER_IP"

# ─── First-time server setup ─────────────────────────────────────
if [ "$SETUP" = "--setup" ]; then
    echo "📦 Running first-time server setup..."
    ssh "$SSH_USER@$SERVER_IP" bash -s <<'SETUP_SCRIPT'
        set -euo pipefail

        # Update system
        apt-get update && apt-get upgrade -y

        # Install Docker
        if ! command -v docker &>/dev/null; then
            curl -fsSL https://get.docker.com | sh
            systemctl enable docker
            systemctl start docker
        fi

        # Install Docker Compose plugin
        if ! docker compose version &>/dev/null; then
            apt-get install -y docker-compose-plugin
        fi

        # Create project directory
        mkdir -p /opt/propsai/deploy/env

        # Set up firewall
        apt-get install -y ufw
        ufw default deny incoming
        ufw default allow outgoing
        ufw allow ssh
        ufw allow 80/tcp
        ufw allow 443/tcp
        ufw --force enable

        echo "✅ Server setup complete"
SETUP_SCRIPT
fi

# ─── Sync project files ──────────────────────────────────────────
echo "📁 Syncing project files..."
rsync -avz --delete \
    --exclude '.git' \
    --exclude 'node_modules' \
    --exclude '.next' \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'deploy/env/backend.env' \
    --exclude 'deploy/env/kalshi.key' \
    --exclude 'backend/.env' \
    --exclude 'frontend/.env.local' \
    --exclude 'backend/app/data/*.db' \
    --exclude 'backend/app/cache' \
    --exclude 'backend/app/models/artifacts' \
    "$PROJECT_ROOT/" "$SSH_USER@$SERVER_IP:$REMOTE_DIR/"

# ─── Check if env files exist on server ──────────────────────────
echo "🔐 Checking environment files..."
ssh "$SSH_USER@$SERVER_IP" bash -s <<'CHECK_ENV'
    if [ ! -f /opt/propsai/deploy/env/backend.env ]; then
        echo "⚠️  WARNING: /opt/propsai/deploy/env/backend.env not found!"
        echo "   Copy the template and fill in your API keys:"
        echo "   scp deploy/env/backend.env root@SERVER_IP:/opt/propsai/deploy/env/backend.env"
        exit 1
    fi
    echo "✅ Environment file found"
CHECK_ENV

# ─── Build and deploy ────────────────────────────────────────────
echo "🐳 Building and deploying containers..."
ssh "$SSH_USER@$SERVER_IP" bash -s <<DEPLOY
    cd /opt/propsai/deploy
    docker compose build --no-cache
    docker compose down
    docker compose up -d
    echo ""
    echo "⏳ Waiting for services to start..."
    sleep 10
    docker compose ps
    echo ""
    echo "✅ Deployment complete!"
    echo "🌐 Dashboard: http://$SERVER_IP"
DEPLOY

echo ""
echo "✅ Deploy finished! Dashboard: http://$SERVER_IP"
