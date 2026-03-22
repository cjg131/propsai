import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

// ── Types (match actual backend responses) ──────────────────────────

export interface DisciplineStatus {
  bankroll: number;
  peak_bankroll: number;
  drawdown_pct: number;
  daily_pnl: number;
  weekly_pnl: number;
  total_deployed: number;
  deployment_pct: number;
  cash_available: number;
  is_halted: boolean;
  halt_reason: string;
  orders_last_hour: number;
  orders_last_day: number;
  positions_count: number;
  strategy_exposure: Record<string, number>;
}

export interface CalibrationBucket {
  count: number;
  avg_predicted: number;
  actual_rate: number;
  calibration_error: number;
}

export interface CalibrationReport {
  total_predictions: number;
  total_pnl: number;
  avg_edge_claimed: number;
  overall_win_rate: number;
  buckets: Record<string, CalibrationBucket>;
  calibration_score: number;
  error?: string;
}

export interface MarketMakerStatus {
  active_markets: number;
  total_inventory: number;
  total_yes: number;
  total_no: number;
  net_delta: number;
  resting_orders: number;
  inventory_by_market: Record<string, { yes: number; no: number }>;
}

// ── Hooks ──────────────────────────────────────────────────────────

export function useDisciplineStatus() {
  return useQuery<DisciplineStatus>({
    queryKey: ["discipline", "status"],
    queryFn: () => api.get("/api/kalshi/agent/discipline/status"),
    refetchInterval: 30_000,
    staleTime: 15_000,
    retry: 1,
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
    retry: 1,
  });
}

export function useMarketMakerStatus() {
  return useQuery<MarketMakerStatus>({
    queryKey: ["discipline", "market-maker-status"],
    queryFn: () => api.get("/api/kalshi/agent/market-maker/status"),
    refetchInterval: 30_000,
    staleTime: 15_000,
    retry: 1,
  });
}
