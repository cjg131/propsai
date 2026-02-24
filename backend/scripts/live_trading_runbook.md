# Live Trading Runbook

## 1) Pre-Flight (must pass)

1. Keep `PAPER_MODE=true` and run:
   - `python3 scripts/live_readiness_check.py`
2. Confirm API health:
   - `GET /health`
   - `GET /api/kalshi/agent/status`
3. Confirm no ghost resting orders:
   - `GET /api/kalshi/agent/trades?status=resting`
4. Confirm monitor heartbeat updates in `/health.guardrails.runtime_health.last_monitor_heartbeat`.

## 2) Secrets Rotation (manual, required before real money)

If any real key/webhook has been shared in plaintext, rotate:

- Kalshi API key pair
- Supabase service key
- OpenAI key
- SportsDataIO key
- Discord webhook URL

Then update deployment env only (do not commit `.env`).

## 3) Canary Launch (recommended)

1. Start with conservative bankroll and limits:
   - `BANKROLL=250`
   - `PAPER_MODE=false`
   - `REQUIRE_WS_FOR_LIVE=true` (optional but safer)
2. Keep auto-kill guards enabled:
   - `AUTO_KILL_ON_ORDER_FAILURES=true`
   - `AUTO_KILL_ON_CANCEL_STORM=false` initially
3. Run for 24h and review:
   - order failures
   - cancel storm warnings
   - realized P&L stability

## 4) Emergency Controls

- Activate kill switch:
  - `POST /api/kalshi/agent/kill-switch` with `{ "active": true }`
- Cancel all resting orders:
  - `POST /api/kalshi/agent/cancel-resting-orders`
- Stop agent loops:
  - `POST /api/kalshi/agent/stop`

## 5) Re-enable after incident

1. Verify `/health` is healthy and runtime flags are green.
2. Verify no unexpected resting orders at broker.
3. Deactivate kill switch:
   - `POST /api/kalshi/agent/kill-switch` with `{ "active": false }`
4. Restart agent in paper first, then live.
