import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

// ── Types ──────────────────────────────────────────────────────────

export interface AgentStatus {
  running: boolean;
  paper_mode: boolean;
  mode?: string;
  kill_switch: boolean;
  bankroll: number;
  effective_bankroll: number;
  strategy_enabled: Record<string, boolean>;
  allowed_live_strategies?: string[];
  allowed_paper_strategies?: string[];
  allocations: Record<string, number>;
  daily_loss_limit: number;
  max_bet_size: number;
  today_pnl: number;
  today_trades: number;
  total_exposure: number;
  max_deployable: number;
  remaining_capital: number;
  over_deployed: boolean;
  odds_api_credits_remaining: number | null;
}

export interface ReadinessCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface ReadinessTarget {
  target_mode: string;
  ready: boolean;
  status: string;
  recommended_start_endpoint: string;
  blockers: string[];
  warnings: string[];
  checks: ReadinessCheck[];
}

export interface AgentReadiness {
  current_mode: string;
  current: ReadinessTarget;
  paper: ReadinessTarget;
  live: ReadinessTarget;
  strategy_policy: {
    allowed_live_strategies: string[];
    allowed_paper_strategies: string[];
    live_weather_only: boolean;
  };
  summary: {
    db_ok: boolean;
    db_failures: string[];
    live_ready: boolean;
    paper_ready: boolean;
  };
}

export interface WeatherVolumeDiagnostics {
  window_days: number;
  funnel: {
    weather_cycles_started: number;
    weather_cycles_completed: number;
    signals_recorded: number;
    trade_attempts: number;
    filled_or_settled_trades: number;
    open_positions_seen: number;
  };
  top_blockers: Array<{
    reason: string;
    stage: string;
    signal_source: string;
    count: number;
  }>;
  rejection_sources: Array<{
    signal_source: string;
    count: number;
  }>;
  top_one_sided_tickers: Array<{
    ticker: string;
    count: number;
  }>;
  daily_rejections: Array<{
    day: string;
    rejections: number;
    near_misses: number;
  }>;
}

export interface WeatherScanStats {
  markets_seen: number;
  hydration_attempted: number;
  detail_updates: number;
  rescued_two_sided_asks: number;
  rescued_full_quotes: number;
  parsed_markets: number;
}

export interface StrategyStats {
  strategy: string;
  total_trades: number;
  wins: number;
  losses: number;
  total_pnl: number;
  total_fees: number;
}

export interface DailyPnl {
  date: string;
  net_pnl: number;
  trades: number;
}

export interface PriceBucketStat {
  bucket: string;
  strategy: string;
  side: string;
  trades: number;
  wins: number;
  total_pnl: number;
}

export interface AgentPerformance {
  bankroll: number;
  paper_mode: boolean;
  kill_switch: boolean;
  strategy_enabled: Record<string, boolean>;
  overall: {
    total_trades: number;
    wins: number;
    losses: number;
    total_pnl: number;
    total_fees: number;
    total_wagered: number;
    win_rate: number;
    roi: number;
  };
  by_strategy: Record<string, StrategyStats>;
  by_price_bucket: PriceBucketStat[];
  daily_pnl: DailyPnl[];
  today_pnl: number;
  today_trades: number;
}

export interface AgentTrade {
  id: string;
  timestamp: string;
  strategy: string;
  ticker: string;
  market_title: string;
  side: string;
  action: string;
  count: number;
  price_cents: number;
  cost: number;
  fee: number;
  order_type: string;
  paper_mode: number;
  order_id: string;
  status: string;
  our_prob: number;
  kalshi_prob: number;
  edge: number;
  signal_source: string;
  result: string;
  pnl: number;
  settled_at: string;
  notes: string;
  thesis: string;
}

export interface AgentSignal {
  id: string;
  timestamp: string;
  strategy: string;
  ticker: string;
  market_title: string;
  side: string;
  our_prob: number;
  kalshi_prob: number;
  edge: number;
  confidence: number;
  recommended_size: number;
  recommended_price: number;
  acted_on: number;
  trade_id: string;
  signal_source: string;
  details: string;
}

export interface AgentLogEntry {
  id: number;
  timestamp: string;
  level: string;
  strategy: string;
  message: string;
  details: string;
}

export interface AgentPosition {
  ticker: string;
  title: string;
  side: string;
  strategy: string;
  signal_source: string;
  contracts: number;
  avg_entry_cents: number;
  total_cost: number;
  total_fees: number;
  max_risk: number;
  max_profit: number;
  avg_our_prob: number;
  avg_entry_kalshi_prob: number;
  avg_entry_edge: number;
  first_entry: string;
  last_entry: string;
  num_fills: number;
  paper_mode: boolean;
  current_yes_ask: number | null;
  current_no_ask: number | null;
  current_yes_bid: number | null;
  current_no_bid: number | null;
  mark_price_cents: number | null;
  unrealized_pnl: number | null;
  current_edge: number | null;
  status: string;
}

export interface PositionsResponse {
  positions: AgentPosition[];
  total: number;
  total_cost: number;
  total_unrealized_pnl: number;
  total_max_risk: number;
}

// ── Hooks ──────────────────────────────────────────────────────────

export function useAgentStatus() {
  return useQuery<AgentStatus>({
    queryKey: ["agent", "status"],
    queryFn: () => api.get("/api/kalshi/agent/status"),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

export function useAgentReadiness() {
  return useQuery<AgentReadiness>({
    queryKey: ["agent", "readiness"],
    queryFn: () => api.get("/api/kalshi/agent/readiness"),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

export function useWeatherVolumeDiagnostics(days = 7) {
  return useQuery<WeatherVolumeDiagnostics>({
    queryKey: ["agent", "weather-volume", days],
    queryFn: () => api.get(`/api/kalshi/agent/weather-volume?days=${days}`),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
}

export function useWeatherScanStats() {
  return useQuery<WeatherScanStats>({
    queryKey: ["agent", "weather-scan-stats"],
    queryFn: () => api.get("/api/kalshi/agent/weather-scan-stats"),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
}

export function useAgentPerformance() {
  return useQuery<AgentPerformance>({
    queryKey: ["agent", "performance"],
    queryFn: () => api.get("/api/kalshi/agent/performance"),
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}

export function useAgentTrades(params?: {
  strategy?: string;
  status?: string;
  limit?: number;
}) {
  const searchParams = new URLSearchParams();
  if (params?.strategy) searchParams.set("strategy", params.strategy);
  if (params?.status) searchParams.set("status", params.status);
  if (params?.limit) searchParams.set("limit", String(params.limit));
  const qs = searchParams.toString();

  return useQuery<{ trades: AgentTrade[]; total: number }>({
    queryKey: ["agent", "trades", params],
    queryFn: () => api.get(`/api/kalshi/agent/trades${qs ? `?${qs}` : ""}`),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
}

export function useAgentSignals(params?: {
  strategy?: string;
  acted_on?: boolean;
  limit?: number;
}) {
  const searchParams = new URLSearchParams();
  if (params?.strategy) searchParams.set("strategy", params.strategy);
  if (params?.acted_on !== undefined)
    searchParams.set("acted_on", String(params.acted_on));
  if (params?.limit) searchParams.set("limit", String(params.limit));
  const qs = searchParams.toString();

  return useQuery<{ signals: AgentSignal[]; total: number }>({
    queryKey: ["agent", "signals", params],
    queryFn: () => api.get(`/api/kalshi/agent/signals${qs ? `?${qs}` : ""}`),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
}

export function useAgentLog(params?: {
  strategy?: string;
  limit?: number;
}) {
  const searchParams = new URLSearchParams();
  if (params?.strategy) searchParams.set("strategy", params.strategy);
  if (params?.limit) searchParams.set("limit", String(params.limit));
  const qs = searchParams.toString();

  return useQuery<{ log: AgentLogEntry[]; total: number }>({
    queryKey: ["agent", "log", params],
    queryFn: () => api.get(`/api/kalshi/agent/log${qs ? `?${qs}` : ""}`),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

export function useAgentPositions() {
  return useQuery<PositionsResponse>({
    queryKey: ["agent", "positions"],
    queryFn: () => api.get("/api/kalshi/agent/positions"),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
}

// ── Mutations ─────────────────────────────────────────────────────

export function useStartAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/kalshi/agent/start"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useStartLiveWeatherAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/kalshi/agent/start-live-weather"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useStopAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/kalshi/agent/stop"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useSetKillSwitch() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (active: boolean) =>
      api.post("/api/kalshi/agent/kill-switch", { active }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useSetPaperMode() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (enabled: boolean) =>
      api.post("/api/kalshi/agent/paper-mode", { enabled }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useToggleStrategy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      strategy,
      enabled,
    }: {
      strategy: string;
      enabled: boolean;
    }) => api.post(`/api/kalshi/agent/strategy/${strategy}/toggle`, { enabled }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useRunWeatherCycle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/kalshi/agent/run/weather"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useRunSportsCycle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/kalshi/agent/run/sports"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useRunCryptoCycle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/kalshi/agent/run/crypto"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useRunFinanceCycle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/kalshi/agent/run/finance"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useRunEconCycle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/kalshi/agent/run/econ"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useRunNbaPropsCycle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/kalshi/agent/run/nba-props"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}

export function useSetReduceOnly() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (enabled: boolean) =>
      api.post("/api/kalshi/agent/reduce-only", { enabled }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent"] });
    },
  });
}
