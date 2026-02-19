#!/bin/bash
# Flip the Kalshi agent from paper trading to LIVE money.
# Usage: ./deploy/go-live.sh [VPS_IP]
# Default VPS: 178.156.176.114

set -e

VPS_IP="${1:-178.156.176.114}"
ENV_FILE="/opt/propsai/deploy/env/backend.env"

echo "🔴 GOING LIVE — flipping PAPER_MODE=false on $VPS_IP"
echo ""
echo "Current env settings:"
ssh root@"$VPS_IP" "grep -E 'PAPER_MODE|BANKROLL' $ENV_FILE"
echo ""

read -p "⚠️  This will place REAL orders on Kalshi with real money. Type 'GOLIVE' to confirm: " confirm
if [ "$confirm" != "GOLIVE" ]; then
    echo "Aborted."
    exit 1
fi

# Flip PAPER_MODE to false
ssh root@"$VPS_IP" "sed -i 's/^PAPER_MODE=.*/PAPER_MODE=false/' $ENV_FILE"

echo ""
echo "Updated env:"
ssh root@"$VPS_IP" "grep -E 'PAPER_MODE|BANKROLL' $ENV_FILE"

# Restart only the backend container (no rebuild needed)
echo ""
echo "Restarting backend container..."
ssh root@"$VPS_IP" "cd /opt/propsai && docker compose -f deploy/docker-compose.yml restart backend"

# Wait and confirm healthy
echo "Waiting for backend to become healthy..."
sleep 10
ssh root@"$VPS_IP" "docker ps --filter name=propsai-backend --format 'Status: {{.Status}}'"

echo ""
echo "✅ LIVE MODE ACTIVE — agent is now placing real Kalshi orders"
echo "📊 Dashboard: http://$VPS_IP/kalshi/agent"
echo ""
echo "To revert to paper mode at any time:"
echo "  ssh root@$VPS_IP \"sed -i 's/^PAPER_MODE=.*/PAPER_MODE=true/' $ENV_FILE && docker compose -f /opt/propsai/deploy/docker-compose.yml restart backend\""
