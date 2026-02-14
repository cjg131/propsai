import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface BetCreate {
  player_id: string;
  player_name: string;
  prop_type: string;
  line: number;
  bet_type: string;
  odds: number;
  sportsbook: string;
  stake: number;
  prediction_id?: string;
  notes?: string;
}

export interface Bet {
  id: string;
  player_id: string;
  player_name: string;
  team: string;
  opponent: string;
  game_date: string;
  prop_type: string;
  line: number;
  bet_type: string;
  odds: number;
  sportsbook: string;
  stake: number;
  status: string;
  actual_value: number | null;
  profit: number | null;
  prediction_id: string | null;
  confidence_tier: number | null;
  notes: string | null;
  created_at: string | null;
  resolved_at: string | null;
}

export interface BetSummary {
  total_bets: number;
  wins: number;
  losses: number;
  pending: number;
  pushes: number;
  win_rate: number;
  roi: number;
  total_wagered: number;
  total_profit: number;
  current_streak: number;
  streak_type: string;
  best_prop_type: string | null;
  worst_prop_type: string | null;
}

export function useBets(filters?: {
  status?: string;
  prop_type?: string;
  limit?: number;
  offset?: number;
}) {
  const params = new URLSearchParams();
  if (filters?.status) params.set("status", filters.status);
  if (filters?.prop_type) params.set("prop_type", filters.prop_type);
  if (filters?.limit) params.set("limit", String(filters.limit));
  if (filters?.offset) params.set("offset", String(filters.offset));
  const qs = params.toString();

  return useQuery({
    queryKey: ["bets", filters],
    queryFn: () =>
      api.get<{ bets: Bet[]; total: number }>(
        `/api/bets${qs ? `?${qs}` : ""}`
      ),
  });
}

export function useBetSummary() {
  return useQuery({
    queryKey: ["bets", "summary"],
    queryFn: () => api.get<BetSummary>("/api/bets/summary"),
  });
}

export function useCreateBet() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (bet: BetCreate) => api.post("/api/bets", bet),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["bets"] });
    },
  });
}
