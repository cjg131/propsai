"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Layers,
  Brain,
  Plus,
  Trash2,
  Sparkles,
  Link2,
  TrendingUp,
  RefreshCw,
  BarChart3,
} from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { useTodayPredictions, type Prediction } from "@/lib/hooks/use-predictions";
import { useCorrelations, type CorrelatedParlay } from "@/lib/hooks/use-data";

const PROP_LABELS: Record<string, string> = {
  points: "PTS", rebounds: "REB", assists: "AST",
  threes: "3PM", steals: "STL", blocks: "BLK", turnovers: "TO",
};

export default function ParlaysPage() {
  const { data } = useTodayPredictions({});
  const { data: corrData, isLoading: corrLoading } = useCorrelations();
  const predictions = useMemo(() => data?.predictions ?? [], [data]);
  const correlatedParlays = useMemo(() => corrData?.correlated_parlays ?? [], [corrData]);
  const [selectedLegs, setSelectedLegs] = useState<string[]>([]);

  const legs = predictions.filter((p) => selectedLegs.includes(p.id));

  const combinedOdds = legs.reduce((acc, leg) => {
    const decimal = leg.best_odds > 0
      ? leg.best_odds / 100 + 1
      : 100 / Math.abs(leg.best_odds) + 1;
    return acc * decimal;
  }, 1);

  const americanCombined = combinedOdds >= 2
    ? Math.round((combinedOdds - 1) * 100)
    : Math.round(-100 / (combinedOdds - 1));

  const toggleLeg = (id: string) => {
    setSelectedLegs((prev) =>
      prev.includes(id) ? prev.filter((l) => l !== id) : [...prev, id]
    );
  };

  // Helper: convert American odds to decimal
  const toDecimal = (odds: number) =>
    odds > 0 ? odds / 100 + 1 : 100 / Math.abs(odds) + 1;

  const toAmerican = (decimal: number) =>
    decimal >= 2
      ? Math.round((decimal - 1) * 100)
      : Math.round(-100 / (decimal - 1));

  // AI suggestions: prioritize high-edge picks, prefer cross-game combos
  const aiSuggestions = useMemo(() => {
    if (predictions.length < 2) return [];

    const scored = [...predictions]
      .filter((p) => p.confidence_tier >= 3 && Math.abs(p.edge_pct) >= 3)
      .map((p) => ({
        ...p,
        score: Math.abs(p.edge_pct) * 0.6 + p.confidence_score * 0.4,
      }))
      .sort((a, b) => b.score - a.score);

    if (scored.length < 2) return [];

    const byGame: Record<string, typeof scored> = {};
    for (const p of scored) {
      const gid = p.game_id || "unknown";
      (byGame[gid] ??= []).push(p);
    }
    const gameIds = Object.keys(byGame);

    type Suggestion = {
      legs: Prediction[];
      avgConf: number;
      avgEdge: number;
      parlayDecimal: number;
      parlayAmerican: number;
    };

    const buildSuggestion = (legs: Prediction[]): Suggestion => {
      const avgConf = legs.reduce((s, l) => s + l.confidence_score, 0) / legs.length;
      const avgEdge = legs.reduce((s, l) => s + Math.abs(l.edge_pct), 0) / legs.length;
      const parlayDecimal = legs.reduce((acc, l) => acc * toDecimal(l.best_odds || -110), 1);
      return { legs, avgConf, avgEdge, parlayDecimal, parlayAmerican: toAmerican(parlayDecimal) };
    };

    const suggestions: Suggestion[] = [];
    const usedKeys = new Set<string>();
    const legKey = (legs: Prediction[]) => legs.map((l) => l.id).sort().join("|");

    if (gameIds.length >= 2) {
      for (let g1 = 0; g1 < gameIds.length; g1++) {
        for (let g2 = g1 + 1; g2 < gameIds.length; g2++) {
          const pool1 = byGame[gameIds[g1]].slice(0, 3);
          const pool2 = byGame[gameIds[g2]].slice(0, 3);
          for (const a of pool1) {
            for (const b of pool2) {
              if (a.player_name === b.player_name) continue;
              const key = legKey([a, b]);
              if (!usedKeys.has(key)) { usedKeys.add(key); suggestions.push(buildSuggestion([a, b])); }
            }
          }
        }
      }
    }

    for (const gid of gameIds) {
      const pool = byGame[gid].slice(0, 4);
      for (let i = 0; i < pool.length; i++) {
        for (let j = i + 1; j < pool.length; j++) {
          if (pool[i].player_name === pool[j].player_name) continue;
          const key = legKey([pool[i], pool[j]]);
          if (!usedKeys.has(key)) { usedKeys.add(key); suggestions.push(buildSuggestion([pool[i], pool[j]])); }
        }
      }
    }

    return suggestions
      .sort((a, b) => (b.avgEdge * 0.6 + b.avgConf * 0.4) - (a.avgEdge * 0.6 + a.avgConf * 0.4))
      .slice(0, 8);
  }, [predictions]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Parlay Builder</h1>
        <p className="text-muted-foreground">
          Build manual parlays or get data-backed correlated prop combos
        </p>
      </div>

      <Tabs defaultValue="correlated" className="space-y-4">
        <TabsList>
          <TabsTrigger value="correlated" className="gap-1.5">
            <Link2 className="h-3.5 w-3.5" />
            Correlated Props
          </TabsTrigger>
          <TabsTrigger value="ai" className="gap-1.5">
            <Sparkles className="h-3.5 w-3.5" />
            AI Suggestions
          </TabsTrigger>
          <TabsTrigger value="manual" className="gap-1.5">
            <Layers className="h-3.5 w-3.5" />
            Manual Builder
          </TabsTrigger>
        </TabsList>

        {/* Correlated Props Tab — powered by real correlation engine */}
        <TabsContent value="correlated">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Data-Backed Correlated Parlays
                {corrData && (
                  <Badge variant="secondary" className="text-xs ml-auto">
                    {corrData.players_with_correlations} players analyzed
                  </Badge>
                )}
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Same-game parlays where props are historically correlated (Pearson r). When one prop hits, the other is statistically more likely to hit too.
              </p>
            </CardHeader>
            <CardContent>
              {corrLoading ? (
                <div className="flex h-[300px] items-center justify-center">
                  <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : correlatedParlays.length === 0 ? (
                <div className="flex h-[300px] items-center justify-center rounded-lg border border-dashed">
                  <div className="text-center">
                    <Link2 className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
                    <p className="text-lg font-medium text-muted-foreground">
                      No correlated parlays found
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Generate predictions first, then correlated parlays will appear
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {correlatedParlays.map((parlay: CorrelatedParlay, idx: number) => (
                    <div
                      key={idx}
                      className="rounded-lg border p-4 hover:bg-muted/30 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-sm">{parlay.player_name}</span>
                          <Badge variant="outline" className="text-xs">
                            Same-Game Parlay
                          </Badge>
                          {parlay.parlay_type === "same_game_correlated" && (
                            <Badge className="text-[10px] bg-green-500/10 text-green-600 border-green-500/20" variant="outline">
                              Positively Correlated
                            </Badge>
                          )}
                          {parlay.parlay_type === "anti_correlated" && (
                            <Badge className="text-[10px] bg-blue-500/10 text-blue-600 border-blue-500/20" variant="outline">
                              Anti-Correlated
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-xs">
                          <span className="text-muted-foreground">
                            r = <span className="font-mono font-bold">{parlay.correlation.toFixed(2)}</span>
                          </span>
                          {parlay.historical_hit_pct > 0 && (
                            <span className="text-muted-foreground">
                              Hit: <span className="font-mono font-bold text-green-600">{(parlay.historical_hit_pct * 100).toFixed(0)}%</span>
                            </span>
                          )}
                          <span className="text-muted-foreground">
                            n = <span className="font-mono">{parlay.sample_size}</span>
                          </span>
                        </div>
                      </div>

                      {/* Correlation strength bar */}
                      <div className="mb-3">
                        <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
                          <span>Correlation Strength</span>
                          <span>{(Math.abs(parlay.correlation) * 100).toFixed(0)}%</span>
                        </div>
                        <Progress value={Math.abs(parlay.correlation) * 100} className="h-1" />
                      </div>

                      <div className="space-y-2">
                        {parlay.legs.map((leg, legIdx) => (
                          <div
                            key={legIdx}
                            className="flex items-center justify-between text-sm bg-muted/30 rounded px-3 py-2"
                          >
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-[10px] font-bold px-1.5 py-0">
                                {PROP_LABELS[leg.prop_type] || leg.prop_type.toUpperCase()}
                              </Badge>
                              <span className={`text-xs font-bold ${leg.bet === "over" ? "text-emerald-500" : "text-red-500"}`}>
                                {leg.bet.toUpperCase()} {leg.line}
                              </span>
                            </div>
                            <div className="flex items-center gap-3 text-xs text-muted-foreground">
                              <span>Pred: <span className="font-mono font-medium">{leg.predicted?.toFixed(1)}</span></span>
                              <span>Conf: <span className="font-mono font-medium">{leg.confidence?.toFixed(0)}%</span></span>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Explanation */}
                      <div className="mt-3 text-[11px] text-muted-foreground bg-muted/20 rounded p-2">
                        <TrendingUp className="h-3 w-3 inline mr-1" />
                        {parlay.parlay_type === "same_game_correlated" ? (
                          <>
                            When {parlay.player_name}&apos;s {PROP_LABELS[parlay.legs[0]?.prop_type] || parlay.legs[0]?.prop_type} goes over,
                            their {PROP_LABELS[parlay.legs[1]?.prop_type] || parlay.legs[1]?.prop_type} also goes over{" "}
                            <strong>{(parlay.historical_hit_pct * 100).toFixed(0)}%</strong> of the time
                            (based on {parlay.sample_size} games, r={parlay.correlation.toFixed(2)}).
                          </>
                        ) : (
                          <>
                            {parlay.player_name}&apos;s {PROP_LABELS[parlay.legs[0]?.prop_type] || parlay.legs[0]?.prop_type} and{" "}
                            {PROP_LABELS[parlay.legs[1]?.prop_type] || parlay.legs[1]?.prop_type} tend to move in opposite directions
                            (r={parlay.correlation.toFixed(2)}, {parlay.sample_size} games).
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ai">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                AI-Suggested Parlays
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Cross-game parlays built from highest-edge predictions
              </p>
            </CardHeader>
            <CardContent>
              {aiSuggestions.length === 0 ? (
                <div className="flex h-[350px] items-center justify-center rounded-lg border border-dashed">
                  <div className="text-center">
                    <Sparkles className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
                    <p className="text-lg font-medium text-muted-foreground">
                      No suggestions available
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Load predictions to see AI-suggested correlated parlays
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {aiSuggestions.map((suggestion, idx) => (
                    <div
                      key={idx}
                      className="rounded-lg border p-4 hover:bg-muted/30 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            {suggestion.legs.length}-Leg Parlay
                          </Badge>
                          <Badge variant="secondary" className="text-xs font-mono">
                            {suggestion.parlayAmerican > 0 ? `+${suggestion.parlayAmerican}` : suggestion.parlayAmerican}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-3 text-xs">
                          <span className="text-muted-foreground">
                            Conf: <span className="font-mono font-medium">{suggestion.avgConf.toFixed(1)}%</span>
                          </span>
                          <span className="text-muted-foreground">
                            Edge: <span className="font-mono font-medium text-green-600">{suggestion.avgEdge.toFixed(1)}%</span>
                          </span>
                          <span className="text-muted-foreground">
                            $10 → <span className="font-mono font-medium text-green-600">${(10 * suggestion.parlayDecimal).toFixed(2)}</span>
                          </span>
                        </div>
                      </div>
                      <div className="space-y-2">
                        {suggestion.legs.map((leg) => (
                          <div
                            key={leg.id}
                            className="flex items-center justify-between text-sm"
                          >
                            <div>
                              <span className="font-medium">{leg.player_name}</span>
                              <span className="text-muted-foreground/60 text-xs ml-1">
                                {leg.team}
                              </span>
                              <span className="text-muted-foreground ml-2">
                                {leg.prop_type} {leg.recommended_bet.toUpperCase()} {leg.line}
                              </span>
                            </div>
                            <span className="font-mono text-xs">
                              {leg.best_odds > 0 ? `+${leg.best_odds}` : leg.best_odds}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="manual">
          <div className="grid gap-6 lg:grid-cols-[1fr_350px]">
            {/* Pick selector */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Available Picks</CardTitle>
              </CardHeader>
              <CardContent>
                {predictions.length === 0 ? (
                  <div className="flex h-[300px] items-center justify-center rounded-lg border border-dashed">
                    <p className="text-sm text-muted-foreground">
                      Load predictions to build parlays
                    </p>
                  </div>
                ) : (
                  <div className="rounded-md border overflow-x-auto max-h-[500px] overflow-y-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[40px]"></TableHead>
                          <TableHead>Player</TableHead>
                          <TableHead>Prop</TableHead>
                          <TableHead className="text-center">Pick</TableHead>
                          <TableHead className="text-center">Odds</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {predictions.map((p) => (
                          <TableRow
                            key={p.id}
                            className={`cursor-pointer ${
                              selectedLegs.includes(p.id) ? "bg-primary/10" : ""
                            }`}
                            onClick={() => toggleLeg(p.id)}
                          >
                            <TableCell>
                              <div
                                className={`h-4 w-4 rounded border ${
                                  selectedLegs.includes(p.id)
                                    ? "bg-primary border-primary"
                                    : "border-muted-foreground/30"
                                }`}
                              />
                            </TableCell>
                            <TableCell className="text-sm font-medium">
                              {p.player_name}
                            </TableCell>
                            <TableCell>
                              <Badge variant="outline" className="text-xs capitalize">
                                {p.prop_type}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-center text-xs">
                              {p.recommended_bet.toUpperCase()} {p.line}
                            </TableCell>
                            <TableCell className="text-center font-mono text-xs">
                              {p.best_odds > 0 ? `+${p.best_odds}` : p.best_odds}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Parlay slip */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Layers className="h-4 w-4" />
                  Parlay Slip
                  {legs.length > 0 && (
                    <Badge variant="secondary" className="text-xs">
                      {legs.length} legs
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {legs.length === 0 ? (
                  <div className="text-center py-8 text-sm text-muted-foreground">
                    <Plus className="h-8 w-8 mx-auto mb-2 text-muted-foreground/50" />
                    Click picks to add legs
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      {legs.map((leg) => (
                        <div
                          key={leg.id}
                          className="flex items-center justify-between p-2 rounded bg-muted/50 text-sm"
                        >
                          <div className="flex-1 min-w-0">
                            <div className="font-medium truncate">{leg.player_name}</div>
                            <div className="text-xs text-muted-foreground">
                              {leg.prop_type} {leg.recommended_bet.toUpperCase()} {leg.line}
                            </div>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0 shrink-0"
                            onClick={() => toggleLeg(leg.id)}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      ))}
                    </div>

                    <div className="border-t pt-3 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Combined Odds</span>
                        <span className="font-mono font-semibold">
                          {americanCombined > 0 ? `+${americanCombined}` : americanCombined}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Payout ($10)</span>
                        <span className="font-mono font-semibold text-green-500">
                          ${(10 * combinedOdds).toFixed(2)}
                        </span>
                      </div>
                    </div>

                    <Button
                      className="w-full"
                      variant="outline"
                      onClick={() => setSelectedLegs([])}
                    >
                      Clear Slip
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
