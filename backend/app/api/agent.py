"""
API endpoints for the Kalshi autonomous trading agent.
Provides status, control, signals, trades, and performance data.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.logging_config import get_logger
from app.services.event_bus import get_event_bus
from app.services.kalshi_agent import get_kalshi_agent
from app.services.live_readiness import evaluate_readiness
from app.services.trading_engine import get_trading_engine

logger = get_logger(__name__)

router = APIRouter()


LIVE_MONITOR_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kalshi Agent Monitor</title>
  <style>
    :root {
      --bg: #f4efe4;
      --panel: #fffaf0;
      --ink: #1e2a24;
      --muted: #5d6a63;
      --accent: #1f7a5c;
      --accent-2: #c96d42;
      --line: #d8cfbf;
      --good: #18794e;
      --bad: #b9382f;
      --warn: #9a6700;
      --shadow: 0 16px 40px rgba(61, 48, 28, 0.08);
      --mono: "SFMono-Regular", "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
      --sans: "Avenir Next", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(201, 109, 66, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(31, 122, 92, 0.14), transparent 30%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
    }
    .shell {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 20px;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.3rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .hero p {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 680px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 250, 240, 0.9);
      padding: 10px 14px;
      font-size: 0.95rem;
      box-shadow: var(--shadow);
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--warn);
      box-shadow: 0 0 0 6px rgba(154, 103, 0, 0.12);
    }
    .dot.live { background: var(--good); box-shadow: 0 0 0 6px rgba(24, 121, 78, 0.12); }
    .dot.paper { background: var(--accent-2); box-shadow: 0 0 0 6px rgba(201, 109, 66, 0.12); }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 16px;
    }
    .card {
      background: rgba(255, 250, 240, 0.94);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .card header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 16px 18px 10px;
    }
    .card h2 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .card .meta {
      color: var(--muted);
      font-size: 0.85rem;
    }
    .content {
      padding: 0 18px 18px;
    }
    .metrics {
      grid-column: span 12;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
    }
    .metric {
      padding: 18px;
      background: rgba(255, 255, 255, 0.45);
      border-radius: 18px;
      border: 1px solid rgba(216, 207, 191, 0.75);
    }
    .metric .label {
      display: block;
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 8px;
    }
    .metric .value {
      font-size: clamp(1.4rem, 3vw, 2.4rem);
      line-height: 1;
      letter-spacing: -0.04em;
    }
    .wide { grid-column: span 7; }
    .mid { grid-column: span 5; }
    .full { grid-column: span 12; }
    .table-wrap, .log-wrap {
      max-height: 420px;
      overflow: auto;
      border-radius: 14px;
      border: 1px solid rgba(216, 207, 191, 0.85);
      background: rgba(255, 255, 255, 0.5);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }
    th, td {
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(216, 207, 191, 0.75);
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      background: #f8f1e4;
      z-index: 1;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .log-line {
      display: grid;
      grid-template-columns: 110px 88px 1fr;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(216, 207, 191, 0.7);
      font-size: 0.9rem;
    }
    .log-time, .mono {
      font-family: var(--mono);
      font-size: 0.82rem;
    }
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(31, 122, 92, 0.12);
      color: var(--accent);
      border: 1px solid rgba(31, 122, 92, 0.18);
    }
    .badge.bad { background: rgba(185, 56, 47, 0.1); color: var(--bad); border-color: rgba(185, 56, 47, 0.16); }
    .badge.warn { background: rgba(154, 103, 0, 0.1); color: var(--warn); border-color: rgba(154, 103, 0, 0.16); }
    .small {
      color: var(--muted);
      font-size: 0.84rem;
    }
    @media (max-width: 1100px) {
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .wide, .mid { grid-column: span 12; }
    }
    @media (max-width: 720px) {
      .shell { padding: 16px; }
      .hero { flex-direction: column; align-items: start; }
      .metrics { grid-template-columns: 1fr; }
      .log-line { grid-template-columns: 1fr; gap: 4px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div>
        <h1>Kalshi Agent Live Monitor</h1>
        <p>Realtime view of live status, open positions, recent trades, and the event stream. This page auto-refreshes and also subscribes to the agent SSE feed.</p>
      </div>
      <div class="status-pill">
        <span id="mode-dot" class="dot"></span>
        <span id="mode-text">Connecting...</span>
      </div>
    </div>

    <div class="metrics">
      <div class="metric card">
        <span class="label">Bankroll</span>
        <div id="metric-bankroll" class="value">$0.00</div>
      </div>
      <div class="metric card">
        <span class="label">Open Positions</span>
        <div id="metric-positions" class="value">0</div>
      </div>
      <div class="metric card">
        <span class="label">Unrealized P&L</span>
        <div id="metric-unrealized" class="value">$0.00</div>
      </div>
      <div class="metric card">
        <span class="label">Recent Trades</span>
        <div id="metric-trades" class="value">0</div>
      </div>
    </div>

    <div class="grid" style="margin-top: 16px;">
      <section class="card wide">
        <header>
          <h2>Open Positions</h2>
          <div id="positions-meta" class="meta">Waiting for data</div>
        </header>
        <div class="content">
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Strategy</th>
                  <th>Side</th>
                  <th>Contracts</th>
                  <th>Cost</th>
                  <th>Unrealized</th>
                </tr>
              </thead>
              <tbody id="positions-body">
                <tr><td colspan="6" class="small">No data yet</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="card mid">
        <header>
          <h2>Status</h2>
          <div id="status-meta" class="meta">Polling every 8s</div>
        </header>
        <div class="content" id="status-body">
          <div class="small">Loading...</div>
        </div>
      </section>

      <section class="card wide">
        <header>
          <h2>Weather Diagnostics</h2>
          <div id="weather-meta" class="meta">Observed-weather gates and provider health</div>
        </header>
        <div class="content" id="weather-body">
          <div class="small">Loading weather diagnostics...</div>
        </div>
      </section>

      <section class="card wide">
        <header>
          <h2>Recent Trades</h2>
          <div id="trades-meta" class="meta">Latest 20</div>
        </header>
        <div class="content">
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Ticker</th>
                  <th>Strategy</th>
                  <th>Side</th>
                  <th>Status</th>
                  <th>P&L</th>
                </tr>
              </thead>
              <tbody id="trades-body">
                <tr><td colspan="6" class="small">No trades yet</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="card mid">
        <header>
          <h2>Live Event Stream</h2>
          <div id="stream-meta" class="meta">Disconnected</div>
        </header>
        <div class="content">
          <div id="events-wrap" class="log-wrap"></div>
        </div>
      </section>

      <section class="card full">
        <header>
          <h2>Agent Log</h2>
          <div class="meta">Latest 80 entries</div>
        </header>
        <div class="content">
          <div id="log-wrap" class="log-wrap"></div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const fmtMoney = (v) => {
      const n = Number(v || 0);
      return `${n < 0 ? '-' : ''}$${Math.abs(n).toFixed(2)}`;
    };
    const fmtTime = (v) => {
      if (!v) return '';
      const d = typeof v === 'number' ? new Date(v * 1000) : new Date(v);
      return isNaN(d.getTime()) ? '' : d.toLocaleTimeString();
    };
    const escapeHtml = (value) => String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');

    function badgeClass(text) {
      const value = String(text || '').toLowerCase();
      if (['error', 'loss', 'failed', 'rejected', 'inactive'].some(k => value.includes(k))) return 'badge bad';
      if (['warning', 'skip', 'pending', 'paper'].some(k => value.includes(k))) return 'badge warn';
      return 'badge';
    }

    function renderStatus(status) {
      const live = !status.paper_mode;
      document.getElementById('mode-dot').className = `dot ${live ? 'live' : 'paper'}`;
      document.getElementById('mode-text').textContent = `${live ? 'Live Mode' : 'Paper Mode'} | ${status.running ? 'Running' : 'Stopped'}`;
      document.getElementById('metric-bankroll').textContent = fmtMoney(status.bankroll);

      const enabled = Object.entries(status.strategy_enabled || {})
        .filter(([, on]) => on)
        .map(([name]) => `<span class="badge">${escapeHtml(name)}</span>`)
        .join(' ');

      document.getElementById('status-body').innerHTML = `
        <div><strong>Kill Switch:</strong> <span class="${badgeClass(status.kill_switch ? 'bad' : 'ok')}">${status.kill_switch ? 'Active' : 'Off'}</span></div>
        <div style="margin-top:8px;"><strong>Enabled:</strong> ${enabled || '<span class="small">None</span>'}</div>
        <div style="margin-top:8px;"><strong>Daily P&L:</strong> ${fmtMoney(status.daily_pnl)}</div>
        <div style="margin-top:8px;"><strong>Exposure:</strong> ${fmtMoney(status.total_exposure)}</div>
        <div style="margin-top:8px;"><strong>Health:</strong> <span class="mono">${escapeHtml(JSON.stringify(status.health || {}))}</span></div>
      `;
    }

    function renderPositions(payload) {
      const positions = payload.positions || [];
      document.getElementById('metric-positions').textContent = String(payload.total || 0);
      document.getElementById('metric-unrealized').textContent = fmtMoney(payload.total_unrealized_pnl);
      document.getElementById('positions-meta').textContent = `${payload.total || 0} open positions | max risk ${fmtMoney(payload.total_max_risk)}`;
      const body = document.getElementById('positions-body');
      if (!positions.length) {
        body.innerHTML = '<tr><td colspan="6" class="small">No open positions</td></tr>';
        return;
      }
      body.innerHTML = positions.map((p) => `
        <tr>
          <td><span class="mono">${escapeHtml(p.ticker)}</span><div class="small">${escapeHtml(p.market_title || '')}</div></td>
          <td>${escapeHtml(p.strategy || '')}</td>
          <td><span class="${badgeClass(p.side)}">${escapeHtml((p.side || '').toUpperCase())}</span></td>
          <td>${escapeHtml(p.contract_count ?? '')}</td>
          <td>${fmtMoney(p.total_cost)}</td>
          <td class="${(p.unrealized_pnl || 0) < 0 ? 'small' : ''}">${fmtMoney(p.unrealized_pnl)}</td>
        </tr>
      `).join('');
    }

    function renderTrades(payload) {
      const trades = payload.trades || [];
      document.getElementById('metric-trades').textContent = String(trades.length);
      const body = document.getElementById('trades-body');
      if (!trades.length) {
        body.innerHTML = '<tr><td colspan="6" class="small">No trades yet</td></tr>';
        return;
      }
      body.innerHTML = trades.slice(0, 20).map((t) => `
        <tr>
          <td class="mono">${fmtTime(t.created_at || t.settled_at)}</td>
          <td class="mono">${escapeHtml(t.ticker || '')}</td>
          <td>${escapeHtml(t.strategy || '')}</td>
          <td><span class="${badgeClass(t.side)}">${escapeHtml((t.side || '').toUpperCase())}</span></td>
          <td><span class="${badgeClass(t.status)}">${escapeHtml(t.status || '')}</span></td>
          <td>${t.pnl == null ? '' : fmtMoney(t.pnl)}</td>
        </tr>
      `).join('');
    }

    function renderWeather(payload) {
      const thresholds = payload.observed_thresholds || {};
      const providers = Object.entries(payload.provider_health || {});
      const summary = payload.near_miss_summary || [];
      const nearMisses = payload.recent_near_misses || [];
      const trades = payload.recent_observed_trades || [];
      const quality = payload.recent_observed_quality || {};

      const providerRows = providers.length ? providers.map(([name, provider]) => {
        const enabled = provider.enabled ? 'on' : 'off';
        const cooldown = Number(provider.cooldown_remaining_sec || 0);
        const cooldownText = cooldown > 0
          ? `${cooldown}s${provider.cooldown_reason ? ` (${provider.cooldown_reason})` : ''}`
          : 'ready';
        return `
          <tr>
            <td>${escapeHtml(name)}</td>
            <td><span class="${badgeClass(enabled)}">${escapeHtml(enabled)}</span></td>
            <td class="mono">${escapeHtml(cooldownText)}</td>
          </tr>
        `;
      }).join('') : '<tr><td colspan="3" class="small">No provider diagnostics</td></tr>';

      const summaryHtml = summary.length ? summary.map((item) => (
        `<span class="badge warn">${escapeHtml(item.reason)}: ${escapeHtml(item.count)}</span>`
      )).join(' ') : '<span class="small">No near misses recorded yet</span>';

      const missRows = nearMisses.length ? nearMisses.slice(0, 6).map((item) => `
        <tr>
          <td class="mono">${fmtTime(item.timestamp)}</td>
          <td class="mono">${escapeHtml(item.ticker || '')}</td>
          <td>${escapeHtml(item.reason || '')}</td>
          <td class="small">${escapeHtml(item.details || '')}</td>
        </tr>
      `).join('') : '<tr><td colspan="4" class="small">No recent near misses</td></tr>';

      const tradeRows = trades.length ? trades.slice(0, 6).map((item) => `
        <tr>
          <td class="mono">${fmtTime(item.timestamp || item.settled_at)}</td>
          <td class="mono">${escapeHtml(item.ticker || '')}</td>
          <td><span class="${badgeClass(item.status)}">${escapeHtml(item.status || '')}</span></td>
          <td>${item.pnl == null ? '' : fmtMoney(item.pnl)}</td>
        </tr>
      `).join('') : '<tr><td colspan="4" class="small">No observed-weather trades yet</td></tr>';

      document.getElementById('weather-body').innerHTML = `
        <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px;">
          <div class="metric">
            <span class="label">Observed Min Edge</span>
            <div class="value">${((thresholds.min_edge || 0) * 100).toFixed(1)}%</div>
          </div>
          <div class="metric">
            <span class="label">Observed Min Price</span>
            <div class="value">${Number(thresholds.min_price_cents || 0).toFixed(0)}c</div>
          </div>
          <div class="metric">
            <span class="label">Observed Quality</span>
            <div class="value">${fmtMoney(quality.avg_pnl || 0)}</div>
            <div class="small">${Number(quality.trades || 0)} trades | ${(Number(quality.win_rate || 0) * 100).toFixed(0)}% win</div>
          </div>
        </div>
        <div style="margin-bottom: 14px;">
          <div class="small" style="margin-bottom: 8px;">Top near-miss reasons</div>
          ${summaryHtml}
        </div>
        <div class="grid" style="margin-top:0;">
          <div class="card mid">
            <header><h2>Providers</h2><div class="meta">Cooldown view</div></header>
            <div class="content">
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Source</th><th>Enabled</th><th>Cooldown</th></tr></thead>
                  <tbody>${providerRows}</tbody>
                </table>
              </div>
            </div>
          </div>
          <div class="card wide">
            <header><h2>Near Misses</h2><div class="meta">Latest 6</div></header>
            <div class="content">
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Time</th><th>Ticker</th><th>Reason</th><th>Details</th></tr></thead>
                  <tbody>${missRows}</tbody>
                </table>
              </div>
            </div>
          </div>
          <div class="card full">
            <header><h2>Observed Trades</h2><div class="meta">Latest live observed-weather trades</div></header>
            <div class="content">
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Time</th><th>Ticker</th><th>Status</th><th>P&amp;L</th></tr></thead>
                  <tbody>${tradeRows}</tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      `;
    }

    function renderLog(payload) {
      const rows = payload.log || [];
      const wrap = document.getElementById('log-wrap');
      wrap.innerHTML = rows.length ? rows.slice(0, 80).map((entry) => `
        <div class="log-line">
          <div class="log-time mono">${fmtTime(entry.created_at)}</div>
          <div><span class="${badgeClass(entry.level)}">${escapeHtml(entry.level || '')}</span></div>
          <div>${escapeHtml(entry.message || '')}</div>
        </div>
      `).join('') : '<div class="log-line"><div class="small">No log entries yet</div></div>';
    }

    function prependEvent(event) {
      const wrap = document.getElementById('events-wrap');
      const line = document.createElement('div');
      line.className = 'log-line';
      line.innerHTML = `
        <div class="log-time mono">${fmtTime(event.timestamp)}</div>
        <div><span class="${badgeClass(event.type)}">${escapeHtml(event.type || '')}</span></div>
        <div class="mono">${escapeHtml(JSON.stringify(event.data || {}))}</div>
      `;
      wrap.prepend(line);
      while (wrap.children.length > 60) {
        wrap.removeChild(wrap.lastChild);
      }
    }

    async function refresh() {
      try {
        const [status, positions, trades, log, recent, weather] = await Promise.all([
          fetch('./status').then(r => r.json()),
          fetch('./positions').then(r => r.json()),
          fetch('./trades?limit=20').then(r => r.json()),
          fetch('./log?limit=80').then(r => r.json()),
          fetch('./events/recent?count=20').then(r => r.json()),
          fetch('./weather-diagnostics').then(r => r.json()),
        ]);
        renderStatus(status);
        renderPositions(positions);
        renderTrades(trades);
        renderWeather(weather);
        renderLog(log);
        const eventsWrap = document.getElementById('events-wrap');
        eventsWrap.innerHTML = '';
        (recent.events || []).slice().reverse().forEach(prependEvent);
      } catch (err) {
        document.getElementById('stream-meta').textContent = `Refresh failed: ${err}`;
      }
    }

    function connectStream() {
      const streamMeta = document.getElementById('stream-meta');
      const source = new EventSource('./stream');
      source.onopen = () => { streamMeta.textContent = 'Connected'; };
      source.onmessage = (evt) => {
        try {
          prependEvent(JSON.parse(evt.data));
        } catch (_err) {}
      };
      source.onerror = () => { streamMeta.textContent = 'Reconnecting...'; };
    }

    refresh();
    connectStream();
    setInterval(refresh, 8000);
  </script>
</body>
</html>
"""


# ── Request models ─────────────────────────────────────────────


class ToggleRequest(BaseModel):
    enabled: bool


class KillSwitchRequest(BaseModel):
    active: bool


class PaperModeRequest(BaseModel):
    enabled: bool


# ── Status & Control ───────────────────────────────────────────


@router.get("/status")
async def get_agent_status():
    """Get current agent status including paper mode, kill switch, bankroll."""
    try:
        agent = get_kalshi_agent()
        return await agent.get_status()
    except Exception as e:
        logger.error("Failed to get agent status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/readiness")
async def get_agent_readiness():
    """Get paper/live readiness with concrete blockers and recommended start paths."""
    try:
        engine = get_trading_engine()
        agent = get_kalshi_agent()
        return evaluate_readiness(engine=engine, agent=agent)
    except Exception as e:
        logger.error("Failed to get readiness", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor", response_class=HTMLResponse)
async def live_monitor():
    """Lightweight live monitor for agent status, positions, trades, and SSE events."""
    return HTMLResponse(LIVE_MONITOR_HTML)


@router.get("/performance")
async def get_agent_performance():
    """Get agent performance summary with P&L, win rate, etc."""
    try:
        engine = get_trading_engine()
        return engine.get_performance_summary()
    except Exception as e:
        logger.error("Failed to get performance", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clv")
async def get_clv_stats(strategy: str = ""):
    """Get Closing Line Value statistics."""
    try:
        engine = get_trading_engine()
        return engine.get_clv_stats(strategy=strategy)
    except Exception as e:
        logger.error("Failed to get CLV stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kill-switch")
async def set_kill_switch(req: KillSwitchRequest):
    """Activate or deactivate the kill switch."""
    engine = get_trading_engine()
    engine.kill_switch = req.active
    engine.log_event(
        "control",
        f"Kill switch {'ACTIVATED' if req.active else 'deactivated'}",
    )
    return {"kill_switch": engine.kill_switch}


@router.post("/paper-mode")
async def set_paper_mode(req: PaperModeRequest):
    """Toggle paper trading mode."""
    engine = get_trading_engine()
    engine.set_paper_mode(req.enabled)
    engine.log_event(
        "control",
        f"Paper mode {'enabled' if req.enabled else 'DISABLED — LIVE TRADING'}",
    )
    return {
        "paper_mode": engine.paper_mode,
        "strategy_enabled": engine.strategy_enabled,
        "allowed_live_strategies": sorted(engine.allowed_live_strategies),
        "allowed_paper_strategies": sorted(engine.allowed_paper_strategies),
    }


@router.post("/strategy/{strategy}/toggle")
async def toggle_strategy(strategy: str, req: ToggleRequest):
    """Enable or disable a specific strategy."""
    engine = get_trading_engine()
    if strategy not in engine.strategy_enabled:
        raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy}")
    if req.enabled:
        allowed, reason = engine.can_enable_strategy(strategy)
        if not allowed:
            raise HTTPException(status_code=409, detail=reason)
    engine.strategy_enabled[strategy] = req.enabled
    engine.log_event(
        "control",
        f"Strategy '{strategy}' {'enabled' if req.enabled else 'disabled'}",
        strategy=strategy,
    )
    return {"strategy": strategy, "enabled": req.enabled}


# ── Agent Start/Stop ───────────────────────────────────────────


@router.post("/start")
async def start_agent():
    """Start the autonomous agent loops in paper mode only."""
    try:
        engine = get_trading_engine()
        if not engine.paper_mode:
            raise HTTPException(
                status_code=409,
                detail="Generic /start is paper-only. Use /start-live-weather for live weather-only trading.",
            )
        agent = get_kalshi_agent()
        await agent.start()
        return {
            "status": "started",
            "paper_mode": agent.engine.paper_mode,
            "mode": "paper",
            "enabled": {k: v for k, v in agent.engine.strategy_enabled.items() if v},
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error("Failed to start agent", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-live-weather")
async def start_live_weather_agent():
    """Start the agent in live weather-only mode within the server process."""
    try:
        agent = get_kalshi_agent()
        engine = get_trading_engine()
        if engine.allowed_live_strategies != {"weather"}:
            raise HTTPException(
                status_code=409,
                detail="Live start is locked to weather-only. Set LIVE_ENABLED_STRATEGIES=weather.",
            )
        engine.set_paper_mode(False)
        engine.sync_bankroll(float(engine.bankroll or 200.0))
        await agent.start()
        return {
            "status": "started",
            "paper_mode": agent.engine.paper_mode,
            "mode": "live_weather_only",
            "enabled": {k: v for k, v in agent.engine.strategy_enabled.items() if v},
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error("Failed to start live weather agent", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_agent():
    """Stop the autonomous agent loops."""
    try:
        agent = get_kalshi_agent()
        await agent.stop()
        return {"status": "stopped"}
    except Exception as e:
        logger.error("Failed to stop agent", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel-resting-orders")
async def cancel_resting_orders():
    """Emergency control: cancel all currently resting live orders and sync DB state."""
    engine = get_trading_engine()
    if engine.paper_mode:
        return {"status": "skipped", "reason": "paper_mode", "canceled": 0}

    canceled = 0
    failed: list[str] = []

    try:
        from app.services.kalshi_api import get_kalshi_client

        client = get_kalshi_client()
        resting = engine.get_resting_trades()
        for trade in resting:
            order_id = trade.get("order_id", "")
            if not order_id or order_id.startswith("PAPER-"):
                continue
            try:
                await client.cancel_order(order_id)
                engine.update_trade_status(trade["id"], "canceled")
                canceled += 1
            except Exception as e:
                failed.append(f"{order_id}: {e}")

        engine.log_event(
            "control",
            f"Manual cancel-resting-orders invoked: canceled={canceled}, failed={len(failed)}",
            strategy="risk",
            details="; ".join(failed[:10]),
        )
        return {"status": "ok", "canceled": canceled, "failed": len(failed), "errors": failed[:10]}
    except Exception as e:
        logger.error("Cancel resting orders failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Manual Triggers ────────────────────────────────────────────


@router.post("/run/weather")
async def run_weather_cycle():
    """Manually trigger one weather strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_weather_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Weather cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/sports")
async def run_sports_cycle():
    """Manually trigger one sports strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_sports_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Sports cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/crypto")
async def run_crypto_cycle():
    """Manually trigger one crypto strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_crypto_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Crypto cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/finance")
async def run_finance_cycle():
    """Manually trigger one finance strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_finance_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Finance cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/econ")
async def run_econ_cycle():
    """Manually trigger one econ strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_econ_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Econ cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/nba-props")
async def run_nba_props_cycle():
    """Manually trigger one NBA props strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_nba_props_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("NBA props cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Data Queries ───────────────────────────────────────────────


@router.get("/positions")
async def get_positions():
    """Get open positions with live market data and unrealized P&L."""
    try:
        agent = get_kalshi_agent()
        positions = await agent.get_positions_with_market_data()
        total_cost = sum(p["total_cost"] for p in positions)
        total_unrealized = sum(p.get("unrealized_pnl") or 0 for p in positions)
        total_max_risk = sum(p["max_risk"] for p in positions)
        return {
            "positions": positions,
            "total": len(positions),
            "total_cost": round(total_cost, 2),
            "total_unrealized_pnl": round(total_unrealized, 2),
            "total_max_risk": round(total_max_risk, 2),
        }
    except Exception as e:
        logger.error("Failed to get positions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/monitor")
async def run_monitor_cycle():
    """Manually trigger one position monitor cycle."""
    try:
        agent = get_kalshi_agent()
        actions = await agent.run_monitor_cycle()
        return {"actions": len(actions), "results": actions}
    except Exception as e:
        logger.error("Monitor cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades")
async def get_trades(
    strategy: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get trade history."""
    engine = get_trading_engine()
    trades = engine.get_trades(strategy=strategy, status=status, limit=limit)
    return {"trades": trades, "total": len(trades)}


@router.get("/signals")
async def get_signals(
    strategy: str | None = Query(None),
    acted_on: bool | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get trading signals."""
    engine = get_trading_engine()
    signals = engine.get_signals(strategy=strategy, acted_on=acted_on, limit=limit)
    return {"signals": signals, "total": len(signals)}


@router.get("/log")
async def get_agent_log(
    strategy: str | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
):
    """Get agent activity log."""
    engine = get_trading_engine()
    log = engine.get_agent_log(limit=limit, strategy=strategy)
    return {"log": log, "total": len(log)}


@router.get("/near-misses")
async def get_near_misses(
    strategy: str = Query("weather"),
    stage: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """Get recent structured candidate rejections and summarized near misses."""
    engine = get_trading_engine()
    rows = engine.get_candidate_rejections(
        strategy=strategy,
        stage=stage,
        near_miss_only=True,
        limit=limit,
    )
    summary = engine.get_near_miss_summary(strategy=strategy, limit=20)
    return {"near_misses": rows, "summary": summary, "total": len(rows)}


@router.get("/weather-diagnostics")
async def get_weather_diagnostics():
    """Get compact observed-weather diagnostics for the live monitor."""
    agent = get_kalshi_agent()
    return agent.get_weather_diagnostics()


@router.get("/weather-volume")
async def get_weather_volume(days: int = Query(7, ge=1, le=30)):
    """Get weather trading funnel and blocker breakdown for the recent window."""
    engine = get_trading_engine()
    return engine.get_weather_volume_diagnostics(days=days)


@router.get("/weather-scan-stats")
async def get_weather_scan_stats():
    """Get latest weather scanner hydration and quote-rescue stats."""
    agent = get_kalshi_agent()
    return agent.scanner.get_weather_scan_stats()


# ── New Endpoints (OpenClaw, Adaptive Thresholds, Signal Stats, Reset) ──


@router.get("/reviews")
async def get_trade_reviews(
    strategy: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """Get GPT-powered trade reviews from OpenClaw."""
    from app.services.trade_analyzer import get_trade_analyzer
    analyzer = get_trade_analyzer()
    reviews = analyzer.get_recent_reviews(strategy=strategy, limit=limit)
    patterns = analyzer.get_pattern_summary()
    return {"reviews": reviews, "patterns": patterns}


@router.get("/thresholds")
async def get_adaptive_thresholds():
    """Get current adaptive thresholds per strategy."""
    from app.services.adaptive_thresholds import get_adaptive_thresholds as _get
    thresholds = _get()
    return thresholds.get_all_thresholds()


@router.get("/signal-stats")
async def get_signal_stats():
    """Get signal component quality stats and dynamic weights."""
    from app.services.signal_scorer import get_signal_scorer
    scorer = get_signal_scorer()
    return scorer.get_all_stats()


@router.get("/backtest/weather")
async def run_weather_backtest_endpoint(days_back: int = Query(default=30, le=365)):
    """Run weather backtesting pipeline on settled weather trades."""
    from app.services.backtesting import run_weather_backtest
    try:
        result = await run_weather_backtest(days_back=days_back)
        return result
    except Exception as e:
        logger.error("Weather backtest failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_paper_trades():
    """Clear all paper trades, signals, and logs for a fresh start."""
    import sqlite3

    from app.services.trading_engine import DB_PATH
    engine = get_trading_engine()
    engine.log_event("control", "DB reset requested via API")

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("DELETE FROM trades WHERE paper_mode = 1")
    c.execute("DELETE FROM signals")
    c.execute("DELETE FROM daily_pnl")
    c.execute("DELETE FROM agent_log")
    conn.commit()
    conn.close()

    engine._first_cycle_done = False
    return {"status": "reset", "message": "Paper trades, signals, and logs cleared"}


@router.get("/stream")
async def agent_event_stream():
    """SSE endpoint — streams real-time agent events (trades, settlements, log entries)."""
    bus = get_event_bus()

    async def event_generator():
        try:
            async for event in bus.subscribe(include_history=True):
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/events/recent")
async def get_recent_events(count: int = Query(default=20, le=50)):
    """Get recent agent events (for initial page load before SSE connects)."""
    bus = get_event_bus()
    return {"events": bus.get_recent(count)}
