import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface ModelContribution {
  model_name: string;
  prediction: number;
  confidence: number;
  weight: number;
}

export interface FeatureImportance {
  feature_name: string;
  importance: number;
  direction: string;
}

export interface Prediction {
  id: string;
  player_id: string;
  player_name: string;
  team: string;
  opponent: string;
  game_id: string;
  prop_type: string;
  line: number;
  predicted_value: number;
  prediction_range_low: number;
  prediction_range_high: number;
  over_probability: number;
  under_probability: number;
  confidence_score: number;
  confidence_tier: number;
  edge_pct: number;
  expected_value: number;
  recommended_bet: string;
  kelly_bet_size: number;
  best_book: string;
  best_odds: number;
  ensemble_agreement: number;
  model_contributions: ModelContribution[];
  feature_importances: FeatureImportance[];
  line_edge_signal: string | null;
  avg_vs_line_pct: number | null;
  pct_games_over_line: number | null;
  l10_avg: number | null;
  created_at: string | null;
}

export interface GameInfo {
  game_id: string;
  home_team: string;
  away_team: string;
  home_team_name: string;
  away_team_name: string;
  game_date: string;
  pick_count: number;
}

export interface PredictionListResponse {
  predictions: Prediction[];
  games: GameInfo[];
  total: number;
  filters_applied: Record<string, string | number | null>;
}

export function useTodayPredictions(filters?: {
  prop_type?: string;
  min_confidence?: number;
  team?: string;
  game_id?: string;
}) {
  const params = new URLSearchParams();
  if (filters?.prop_type) params.set("prop_type", filters.prop_type);
  if (filters?.min_confidence)
    params.set("min_confidence", String(filters.min_confidence));
  if (filters?.team) params.set("team", filters.team);
  if (filters?.game_id) params.set("game_id", filters.game_id);

  const query = params.toString();
  const endpoint = `/api/predictions/today${query ? `?${query}` : ""}`;

  return useQuery<PredictionListResponse>({
    queryKey: ["predictions", "today", filters],
    queryFn: () => api.get<PredictionListResponse>(endpoint),
  });
}

export function usePredictionDetail(predictionId: string) {
  return useQuery({
    queryKey: ["predictions", predictionId],
    queryFn: () => api.get(`/api/predictions/${predictionId}`),
    enabled: !!predictionId,
  });
}

export function useGeneratePredictions() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<{ status: string; predictions_created: number }>("/api/predictions/generate"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}

// ── Backtesting ──────────────────────────────────────────────────────

export interface BacktestConfig {
  period?: string;
  start_date?: string;
  end_date?: string;
  prop_types?: string;
  min_confidence?: number;
  bankroll?: number;
  kelly_fraction?: number;
}

export interface BacktestBet {
  date: string;
  player: string;
  prop_type: string;
  line: number;
  predicted: number;
  actual: number;
  bet: string;
  hit: boolean;
  confidence: number;
  stake: number;
  profit: number;
  bankroll: number;
}

export interface BacktestResult {
  status: string;
  message?: string;
  config?: {
    start_date: string;
    end_date: string;
    prop_types: string[];
    min_confidence: number;
    starting_bankroll: number;
    kelly_fraction: number;
  };
  summary?: {
    total_bets: number;
    wins: number;
    losses: number;
    hit_rate: number;
    total_wagered: number;
    total_profit: number;
    roi: number;
    final_bankroll: number;
    max_drawdown: number;
    game_dates_simulated: number;
    current_streak: number;
    streak_type: string;
  };
  equity_curve?: { date: string; bankroll: number; bet_num: number }[];
  daily_results?: { date: string; bets: number; hits: number; profit: number; bankroll: number }[];
  by_prop_type?: Record<string, { bets: number; hits: number; misses: number; hit_rate: number; profit: number }>;
  calibration?: Record<string, { hit_rate: number; total: number }>;
  bet_log?: BacktestBet[];
  total_bet_log_size?: number;
}

export function useRunBacktest() {
  return useMutation({
    mutationFn: (config: BacktestConfig) => {
      const params = new URLSearchParams();
      if (config.period) params.set("period", config.period);
      if (config.start_date) params.set("start_date", config.start_date);
      if (config.end_date) params.set("end_date", config.end_date);
      if (config.prop_types) params.set("prop_types", config.prop_types);
      if (config.min_confidence) params.set("min_confidence", String(config.min_confidence));
      if (config.bankroll) params.set("bankroll", String(config.bankroll));
      if (config.kelly_fraction) params.set("kelly_fraction", String(config.kelly_fraction));
      const qs = params.toString();
      return api.post<BacktestResult>(`/api/predictions/backtest${qs ? `?${qs}` : ""}`);
    },
  });
}
