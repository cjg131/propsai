import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface AppSettings {
  bankroll: number;
  unit_size: number;
  active_preset: string;
  fantasy_format: string;
  preferred_books: string[];
}

export interface ModelPreset {
  id: string;
  name: string;
  description: string;
  kelly_fraction: number;
  min_confidence: number;
  is_builtin: boolean;
  model_weights: Record<string, number> | null;
}

export function useAppSettings() {
  return useQuery({
    queryKey: ["settings"],
    queryFn: () => api.get<AppSettings>("/api/settings"),
  });
}

export function useUpdateSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (settings: Partial<AppSettings>) =>
      api.put("/api/settings", settings),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });
}

export function usePresets() {
  return useQuery({
    queryKey: ["settings", "presets"],
    queryFn: () => api.get<ModelPreset[]>("/api/settings/presets"),
  });
}

export function useCreatePreset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (preset: {
      name: string;
      description?: string;
      kelly_fraction?: number;
      min_confidence?: number;
    }) => api.post<ModelPreset>("/api/settings/presets", preset),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings", "presets"] });
    },
  });
}
