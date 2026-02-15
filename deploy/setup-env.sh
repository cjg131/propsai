#!/bin/bash
set -euo pipefail

# ─── PropsAI: Copy local env to VPS ──────────────────────────────
# Run from your Mac: ./deploy/setup-env.sh <server-ip>
# This copies your local .env and kalshi.key to the VPS

SERVER_IP="${1:?Usage: ./deploy/setup-env.sh <server-ip>}"
SSH_USER="root"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "🔐 Setting up environment on $SERVER_IP"

# Create env directory on server
ssh "$SSH_USER@$SERVER_IP" "mkdir -p /opt/propsai/deploy/env"

# Copy backend .env (transform for production)
echo "📋 Copying backend .env..."
sed \
    -e "s|APP_ENV=.*|APP_ENV=production|" \
    -e "s|APP_DEBUG=.*|APP_DEBUG=false|" \
    -e "s|CORS_ORIGINS=.*|CORS_ORIGINS=http://$SERVER_IP|" \
    -e "s|KALSHI_PRIVATE_KEY_PATH=.*|KALSHI_PRIVATE_KEY_PATH=/app/kalshi.key|" \
    "$PROJECT_ROOT/backend/.env" | \
    ssh "$SSH_USER@$SERVER_IP" "cat > /opt/propsai/deploy/env/backend.env"

# Copy Kalshi private key if it exists
if [ -f "$PROJECT_ROOT/backend/kalshi.key" ]; then
    echo "🔑 Copying Kalshi private key..."
    scp "$PROJECT_ROOT/backend/kalshi.key" "$SSH_USER@$SERVER_IP:/opt/propsai/deploy/env/kalshi.key"
    ssh "$SSH_USER@$SERVER_IP" "chmod 600 /opt/propsai/deploy/env/kalshi.key"
else
    echo "⚠️  No kalshi.key found at backend/kalshi.key — skipping"
fi

echo ""
echo "✅ Environment files deployed to $SERVER_IP"
echo "   /opt/propsai/deploy/env/backend.env"
echo "   /opt/propsai/deploy/env/kalshi.key"
echo ""
echo "Next step: ./deploy/deploy.sh $SERVER_IP --setup"
