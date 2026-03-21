import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

// ── Types ──────────────────────────────────────────────────────────

export interface CircuitBreakerState {
  daily_limit_hit: boolean;
  daily_loss_limit: number;
  daily_pnl: number;
  weekly_limit_hit: boolean;
  weekly_loss_limit: number;
  weekly_pnl: number;
  drawdown_limit_hit: boolean;
  drawdown_limit_pct: number;
  current_drawdown_pct: number;
}

export interface RateLimits {
  orders_today: number;
  max_orders_daily: number;
  orders_this_hour: number;
  max_orders_hourly: number;
}

export interface Concentration {
  total_deployed_pct: number;
  largest_market_pct: number;
  largest_strategy_pct: number;
}

export interface DisciplineStatus {
  circuit_breaker: CircuitBreakerState;
  rate_limits: RateLimits;
  concentration: Concentration;
  cash_reserve_pct: number;
  timestamp: string;
}

export interface CalibrationBucket {
  bucket_start: number;
  bucket_end: number;
  predicted_prob: number;
  actual_win_rate: number;
  trade_count: number;
}

export interface StrategyCalibrationStat {
  strategy: string;
  total_trades: number;
  win_rate: number;
  avg_edge: number;
  total_pnl: number;
}

export interface CalibrationReport {
  calibration_curve: CalibrationBucket[];
  strategy_stats: StrategyCalibrationStat[];
  overall_win_rate: number;
  overall_edge: number;
  timestamp: string;
}

export interface MarketInventory {
  ticker: string;
  title: string;
  side: string;
  contracts: number;
  position_value: number;
}

export interface MarketMakerStatus {
  active_markets: number;
  total_inventory: number;
  net_delta: number;
  resting_orders: number;
  inventory_by_market: MarketInventory[];
  timestamp: string;
}

// ── Hooks ──────────────────────────────────────────────────────────

export function useDisciplineStatus() {
  return useQuery<DisciplineStatus>({
    queryKey: ["discipline", "status"],
    queryFn: () => api.get("/api/kalshi/agent/discipline/status"),
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}

export function useCalibrationReport(strategy?: string) {
  const searchParams = new URLSearchParams();
  if (strategy) searchParams.set("strategy", strategy);
  const qs = searchParams.toString();

  return useQuery<CalibrationReport>({
    queryKey: ["discipline", "calibration", strategy],
    queryFn: () =>
      api.get(`/api/kalshi/agent/discipline/calibration${qs ? `?${qs}` : ""}`),
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}

export function useMarketMakerStatus() {
  return useQuery<MarketMakerStatus>({
    queryKey: ["discipline", "market-maker-status"],
    queryFn: () => api.get("/api/kalshi/agent/market-maker/status"),
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}
