import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

// ── Types ──────────────────────────────────────────────────────────

export interface KalshiMarket {
  ticker: string;
  event_ticker: string;
  title: string;
  subtitle: string;
  player_name: string;
  prop_type: string;
  line: number | null;
  yes_bid: number;
  yes_ask: number;
  no_bid: number;
  no_ask: number;
  last_price: number;
  implied_prob_over: number;
  implied_prob_under: number;
  volume: number;
  volume_24h: number;
  open_interest: number;
  status: string;
  close_time: string;
  strike_type: string;
}

export interface KalshiEdge extends KalshiMarket {
  edge_type: string | null;
  edge_pct: number;
  our_prob_over: number | null;
  our_prob_under: number | null;
  recommendation: string | null;
}

export interface KalshiMarketsResponse {
  markets: KalshiMarket[];
  total: number;
}

export interface KalshiEdgesResponse {
  edges: KalshiEdge[];
  total: number;
  total_markets_scanned: number;
}

export interface KalshiBalance {
  balance: number;
  payout: number;
}

export interface KalshiPosition {
  ticker: string;
  event_ticker: string;
  market_ticker: string;
  side: string;
  quantity: number;
  average_price: number;
  market_title: string;
}

export interface KalshiPositionsResponse {
  market_positions: KalshiPosition[];
  cursor: string;
}

// ── Hooks ──────────────────────────────────────────────────────────

export function useKalshiMarkets() {
  return useQuery<KalshiMarketsResponse>({
    queryKey: ["kalshi", "markets"],
    queryFn: () => api.get("/api/kalshi/markets"),
    refetchInterval: 60_000, // Refresh every 60s
    staleTime: 30_000,
  });
}

export function useKalshiEdges() {
  return useQuery<KalshiEdgesResponse>({
    queryKey: ["kalshi", "edges"],
    queryFn: () => api.get("/api/kalshi/edges"),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

export function useKalshiBalance() {
  return useQuery<KalshiBalance>({
    queryKey: ["kalshi", "balance"],
    queryFn: () => api.get("/api/kalshi/balance"),
    refetchInterval: 120_000,
    staleTime: 60_000,
    retry: false,
  });
}

export function useKalshiPositions() {
  return useQuery<KalshiPositionsResponse>({
    queryKey: ["kalshi", "positions"],
    queryFn: () => api.get("/api/kalshi/positions"),
    refetchInterval: 60_000,
    staleTime: 30_000,
    retry: false,
  });
}

// ── History Types ─────────────────────────────────────────────────

export interface KalshiSettledMarket {
  ticker: string;
  event_ticker: string;
  series_ticker: string;
  title: string;
  subtitle: string;
  player_name: string;
  prop_type: string;
  line: number | null;
  yes_bid: number;
  yes_ask: number;
  no_bid: number;
  no_ask: number;
  last_price: number;
  implied_prob_over: number;
  implied_prob_under: number;
  volume: number;
  volume_24h: number;
  open_interest: number;
  status: string;
  close_time: string;
  settlement_ts: string;
  result: string;
  settlement_value: number;
  strike_type: string;
}

export interface KalshiHistorySummary {
  total_markets: number;
  unique_players: number;
  date_range: { start: string | null; end: string | null };
  game_days: number;
  by_prop_type: Record<string, number>;
  yes_results: number;
  no_results: number;
}

export interface KalshiHistoryResponse {
  markets: KalshiSettledMarket[];
  total: number;
  filtered: number;
  offset: number;
  limit: number;
  summary: KalshiHistorySummary;
}

export interface KalshiBacktestBet {
  ticker: string;
  player: string;
  prop: string;
  line: number;
  direction: string;
  buy_price: number;
  our_prob: number;
  kalshi_prob: number;
  edge: number;
  result: string;
  pnl: number;
  date: string;
  volume: number;
}

export interface KalshiBacktestResponse {
  settings: { min_edge: number; min_volume: number; bet_size: number };
  summary: {
    total_markets: number;
    matched_player: number;
    had_features: number;
    had_edge: number;
    bets_placed: number;
    bets_won: number;
    win_rate: number;
    total_wagered: number;
    total_pnl: number;
    roi: number;
  };
  by_prop: Record<string, {
    bets: number; wins: number; wagered: number;
    pnl: number; roi: number; avg_edge: number;
  }>;
  by_direction: Record<string, {
    bets: number; wins: number; wagered: number; pnl: number; roi: number;
  }>;
  bets: KalshiBacktestBet[];
}

// ── History Hooks ─────────────────────────────────────────────────

export function useKalshiHistory(params?: {
  prop_type?: string;
  player?: string;
  limit?: number;
  offset?: number;
}) {
  const searchParams = new URLSearchParams();
  if (params?.prop_type) searchParams.set("prop_type", params.prop_type);
  if (params?.player) searchParams.set("player", params.player);
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.offset) searchParams.set("offset", String(params.offset));
  const qs = searchParams.toString();

  return useQuery<KalshiHistoryResponse>({
    queryKey: ["kalshi", "history", params],
    queryFn: () => api.get(`/api/kalshi/history${qs ? `?${qs}` : ""}`),
    staleTime: 5 * 60_000,
  });
}

export function useKalshiBacktest() {
  return useQuery<KalshiBacktestResponse>({
    queryKey: ["kalshi", "backtest"],
    queryFn: () => api.get("/api/kalshi/history/backtest"),
    staleTime: 10 * 60_000,
  });
}
