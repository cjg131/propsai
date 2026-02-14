import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface PlayerStats {
  season_avg_points: number;
  season_avg_rebounds: number;
  season_avg_assists: number;
  season_avg_threes: number;
  season_avg_steals: number;
  season_avg_blocks: number;
  season_avg_turnovers: number;
  season_avg_minutes: number;
  last5_avg_points: number;
  last5_avg_rebounds: number;
  last5_avg_assists: number;
  last10_avg_points: number;
  last10_avg_rebounds: number;
  last10_avg_assists: number;
  home_avg_points: number;
  away_avg_points: number;
  usage_rate: number;
  games_played: number;
}

export interface Player {
  id: string;
  name: string;
  team: string;
  team_id: string;
  position: string;
  jersey_number: string | null;
  headshot_url: string | null;
  stats: PlayerStats | null;
  is_starter: boolean;
  is_recently_traded: boolean;
  is_rookie: boolean;
}

export interface ScoutingReport {
  player_id: string;
  report: string;
  generated_at: string | null;
  model_used: string;
  matchup_analysis: string | null;
  injury_impact: string | null;
  prop_recommendations: Record<string, unknown>[];
}

export function useSearchPlayers(query?: string, team?: string) {
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  if (team) params.set("team", team);
  const qs = params.toString();

  return useQuery({
    queryKey: ["players", "search", query, team],
    queryFn: () =>
      api.get<{ players: Player[]; total: number }>(
        `/api/players${qs ? `?${qs}` : ""}`
      ),
    enabled: !!query || !!team,
  });
}

export function usePlayer(playerId: string) {
  return useQuery({
    queryKey: ["players", playerId],
    queryFn: () => api.get<{ player: Player }>(`/api/players/${playerId}`),
    enabled: !!playerId,
  });
}

export function useScoutingReport(playerId: string) {
  return useQuery({
    queryKey: ["players", playerId, "scouting-report"],
    queryFn: () =>
      api.get<ScoutingReport>(`/api/players/${playerId}/scouting-report`),
    enabled: !!playerId,
  });
}
