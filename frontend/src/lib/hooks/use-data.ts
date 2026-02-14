import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface DataStatus {
  last_refresh: string | null;
  total_players: number;
  total_games: number;
  total_seasons: number;
  api_quota_used: number;
  api_quota_limit: number;
  model_last_trained: string | null;
}

export function useDataStatus() {
  return useQuery({
    queryKey: ["data", "status"],
    queryFn: () => api.get<DataStatus>("/api/data/status"),
  });
}

export function useRefreshData() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/data/refresh"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data"] });
      queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}

export function useLoadHistoricalData() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (seasons: number = 10) =>
      api.post(`/api/data/load-historical?seasons=${seasons}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data"] });
    },
  });
}

export function useRetrainModels() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/data/retrain"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data"] });
      queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}

export function useExportData() {
  return useMutation({
    mutationFn: async (format: "csv" | "json") => {
      const resp = await fetch(`/api/data/export/${format}`);
      if (!resp.ok) throw new Error("Export failed");
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `propsai_export.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      return { status: "ok" };
    },
  });
}

export interface NewsArticle {
  title: string;
  summary: string;
  source: string;
  published: string;
  url: string;
}

export interface PlayerSentiment {
  news_sentiment: number;
  news_volume: number;
  injury_mentioned: number;
  rest_mentioned: number;
  trade_mentioned: number;
  hot_streak_mentioned: number;
}

export interface NewsResponse {
  articles: NewsArticle[];
  total_articles: number;
  player_sentiment: Record<string, PlayerSentiment>;
  players_mentioned: number;
  sources: { rss: number; newsapi: number };
}

export function useNewsSentiment() {
  return useQuery<NewsResponse>({
    queryKey: ["data", "news"],
    queryFn: () => api.get<NewsResponse>("/api/data/news"),
    staleTime: 5 * 60 * 1000,
  });
}

export interface CorrelatedParlay {
  player_name: string;
  legs: {
    prop_type: string;
    line: number;
    predicted: number;
    bet: string;
    confidence: number;
  }[];
  correlation: number;
  historical_hit_pct: number;
  combined_confidence: number;
  sample_size: number;
  parlay_type: string;
}

export interface CorrelationsResponse {
  players_with_correlations: number;
  correlated_parlays: CorrelatedParlay[];
  total_suggestions: number;
}

export function useCorrelations() {
  return useQuery<CorrelationsResponse>({
    queryKey: ["data", "correlations"],
    queryFn: () => api.get<CorrelationsResponse>("/api/data/correlations"),
    staleTime: 10 * 60 * 1000,
  });
}

export function useEvaluatePredictions() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (evalDate?: string) =>
      api.post(evalDate ? `/api/predictions/evaluate/${evalDate}` : "/api/predictions/evaluate"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}

export function useSnapshotLines() {
  return useMutation({
    mutationFn: () => api.post("/api/predictions/snapshot-lines"),
  });
}

export function useRefreshOdds() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/odds/refresh"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["odds"] });
      queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}

export function usePrefetchData() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/api/data/prefetch"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data"] });
    },
  });
}
