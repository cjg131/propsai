"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  TrendingUp,
  DollarSign,
  Target,
  BarChart3,
  RefreshCw,
  Star,
  ChevronDown,
  ArrowUpDown,
  Filter,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { useTodayPredictions, useGeneratePredictions, type Prediction, type GameInfo } from "@/lib/hooks/use-predictions";
import { useNewsSentiment, type PlayerSentiment } from "@/lib/hooks/use-data";
import {
  Newspaper,
  AlertTriangle,
  Flame,
  ArrowRightLeft,
  BedDouble,
} from "lucide-react";

const PROP_LABELS: Record<string, string> = {
  points: "PTS",
  rebounds: "REB",
  assists: "AST",
  threes: "3PM",
  three_pointers_made: "3PM",
  steals: "STL",
  blocks: "BLK",
  turnovers: "TO",
};

const PROP_COLORS: Record<string, string> = {
  points: "bg-blue-500/10 text-blue-600 border-blue-500/20",
  rebounds: "bg-orange-500/10 text-orange-600 border-orange-500/20",
  assists: "bg-purple-500/10 text-purple-600 border-purple-500/20",
  threes: "bg-cyan-500/10 text-cyan-600 border-cyan-500/20",
  steals: "bg-pink-500/10 text-pink-600 border-pink-500/20",
  blocks: "bg-amber-500/10 text-amber-600 border-amber-500/20",
  turnovers: "bg-slate-500/10 text-slate-600 border-slate-500/20",
};

const PROP_TYPES = [
  { value: "all", label: "All Props" },
  { value: "points", label: "Points" },
  { value: "rebounds", label: "Rebounds" },
  { value: "assists", label: "Assists" },
  { value: "threes", label: "3-Pointers" },
  { value: "steals", label: "Steals" },
  { value: "blocks", label: "Blocks" },
  { value: "turnovers", label: "Turnovers" },
];

const CONFIDENCE_FILTERS = [
  { value: "all", label: "All Confidence" },
  { value: "40", label: "2+ Stars" },
  { value: "60", label: "3+ Stars" },
  { value: "75", label: "4+ Stars" },
  { value: "90", label: "5 Stars Only" },
];

function ConfidenceStars({ tier }: { tier: number }) {
  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((i) => (
        <Star
          key={i}
          className={`h-3 w-3 ${
            i <= tier
              ? "fill-yellow-400 text-yellow-400"
              : "text-muted-foreground/20"
          }`}
        />
      ))}
    </div>
  );
}

function GameCard({
  game,
  isSelected,
  onClick,
}: {
  game: GameInfo;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <Card
      className={`cursor-pointer transition-all hover:shadow-md ${
        isSelected
          ? "ring-2 ring-primary bg-primary/5 shadow-md"
          : "hover:bg-muted/50"
      }`}
      onClick={onClick}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="text-right min-w-[36px]">
              <div className="text-sm font-bold">{game.away_team}</div>
              <div className="text-[10px] text-muted-foreground leading-tight">{game.away_team_name}</div>
            </div>
            <span className="text-xs text-muted-foreground font-medium">@</span>
            <div className="min-w-[36px]">
              <div className="text-sm font-bold">{game.home_team}</div>
              <div className="text-[10px] text-muted-foreground leading-tight">{game.home_team_name}</div>
            </div>
          </div>
          <Badge variant="secondary" className="text-[10px] font-semibold shrink-0">
            {game.pick_count} picks
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

function SentimentBadges({ sentiment }: { sentiment?: PlayerSentiment }) {
  if (!sentiment || sentiment.news_volume === 0) return null;
  const s = sentiment;
  const sentColor =
    s.news_sentiment > 0.3 ? "text-green-500" :
    s.news_sentiment < -0.3 ? "text-red-500" : "text-yellow-500";
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {s.injury_mentioned === 1 && (
        <Badge variant="outline" className="text-[9px] px-1 py-0 border-red-500/40 text-red-500 gap-0.5">
          <AlertTriangle className="h-2.5 w-2.5" /> INJ
        </Badge>
      )}
      {s.rest_mentioned === 1 && (
        <Badge variant="outline" className="text-[9px] px-1 py-0 border-orange-500/40 text-orange-500 gap-0.5">
          <BedDouble className="h-2.5 w-2.5" /> REST
        </Badge>
      )}
      {s.trade_mentioned === 1 && (
        <Badge variant="outline" className="text-[9px] px-1 py-0 border-purple-500/40 text-purple-500 gap-0.5">
          <ArrowRightLeft className="h-2.5 w-2.5" /> TRADE
        </Badge>
      )}
      {s.hot_streak_mentioned === 1 && (
        <Badge variant="outline" className="text-[9px] px-1 py-0 border-orange-400/40 text-orange-400 gap-0.5">
          <Flame className="h-2.5 w-2.5" /> HOT
        </Badge>
      )}
      {s.news_volume > 0 && s.injury_mentioned === 0 && s.rest_mentioned === 0 && s.trade_mentioned === 0 && s.hot_streak_mentioned === 0 && (
        <Badge variant="outline" className={`text-[9px] px-1 py-0 gap-0.5 ${sentColor}`}>
          <Newspaper className="h-2.5 w-2.5" /> {s.news_volume}
        </Badge>
      )}
    </div>
  );
}

function PredictionCard({ prediction, sentiment }: { prediction: Prediction; sentiment?: PlayerSentiment }) {
  const [expanded, setExpanded] = useState(false);
  const isOver = prediction.recommended_bet === "over";
  const propLabel = PROP_LABELS[prediction.prop_type] || prediction.prop_type.toUpperCase();
  const propColor = PROP_COLORS[prediction.prop_type] || "bg-muted text-foreground";
  const oddsStr = prediction.best_odds > 0 ? `+${prediction.best_odds}` : `${prediction.best_odds}`;

  return (
    <Card
      className="overflow-hidden transition-shadow hover:shadow-md cursor-pointer"
      onClick={() => setExpanded(!expanded)}
    >
      <CardContent className="p-0">
        {/* Main row */}
        <div className="flex items-center gap-3 px-4 py-3">
          {/* Player info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm truncate">{prediction.player_name}</span>
              <span className="text-xs text-muted-foreground shrink-0">{prediction.team}</span>
              {prediction.opponent && (
                <span className="text-[10px] text-muted-foreground shrink-0">vs {prediction.opponent}</span>
              )}
              <SentimentBadges sentiment={sentiment} />
            </div>
            <div className="flex items-center gap-2 mt-1">
              <Badge variant="outline" className={`text-[10px] font-bold px-1.5 py-0 ${propColor}`}>
                {propLabel}
              </Badge>
              <ConfidenceStars tier={prediction.confidence_tier} />
            </div>
          </div>

          {/* Line & Prediction */}
          <div className="text-center shrink-0 w-20">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Line</div>
            <div className="font-mono text-sm font-medium">{prediction.line.toFixed(1)}</div>
          </div>

          <div className="text-center shrink-0 w-20">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Pred</div>
            <div className="font-mono text-sm font-bold">{prediction.predicted_value.toFixed(1)}</div>
          </div>

          {/* Pick badge */}
          <div className="shrink-0 flex items-center gap-1">
            {((prediction.line_edge_signal === "strong_over" && ["threes", "points"].includes(prediction.prop_type)) ||
              prediction.line_edge_signal === "strong_under") && (
              <Badge className={`text-[9px] font-bold px-1.5 py-0.5 text-white border-0 animate-pulse ${
                prediction.line_edge_signal === "strong_over" ? "bg-amber-500 hover:bg-amber-600" : "bg-violet-500 hover:bg-violet-600"
              }`}>
                EDGE
              </Badge>
            )}
            <Badge
              className={`text-xs font-bold px-3 py-1 ${
                isOver
                  ? "bg-emerald-500 hover:bg-emerald-600 text-white border-0"
                  : "bg-red-500 hover:bg-red-600 text-white border-0"
              }`}
            >
              {isOver ? "↑" : "↓"} {prediction.recommended_bet.toUpperCase()}
            </Badge>
          </div>

          {/* Edge */}
          <div className="text-center shrink-0 w-16">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Edge</div>
            <div className={`font-mono text-sm font-semibold ${
              prediction.edge_pct > 5 ? "text-emerald-500" : prediction.edge_pct > 2 ? "text-green-500" : "text-muted-foreground"
            }`}>
              +{prediction.edge_pct.toFixed(1)}%
            </div>
          </div>

          {/* Odds */}
          <div className="text-center shrink-0 w-20 hidden sm:block">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">{prediction.best_book}</div>
            <div className={`font-mono text-sm ${prediction.best_odds > 0 ? "text-emerald-500" : "text-muted-foreground"}`}>
              {oddsStr}
            </div>
          </div>

          {/* Wager */}
          <div className="text-center shrink-0 w-14 hidden md:block">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Wager</div>
            <div className="font-mono text-sm font-medium">
              {prediction.kelly_bet_size > 0 ? `$${prediction.kelly_bet_size.toFixed(0)}` : "—"}
            </div>
          </div>

          <ChevronDown className={`h-4 w-4 text-muted-foreground shrink-0 transition-transform ${expanded ? "rotate-180" : ""}`} />
        </div>

        {/* Expanded details */}
        {expanded && (
          <div className="border-t bg-muted/30 px-4 py-3">
            <div className="grid gap-4 sm:grid-cols-3">
              <div>
                <div className="text-xs font-semibold text-muted-foreground mb-1.5">Prediction Range</div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="font-mono">{prediction.prediction_range_low.toFixed(1)}</span>
                  <Progress
                    value={
                      ((prediction.predicted_value - prediction.prediction_range_low) /
                        Math.max(prediction.prediction_range_high - prediction.prediction_range_low, 0.1)) * 100
                    }
                    className="h-1.5 flex-1"
                  />
                  <span className="font-mono">{prediction.prediction_range_high.toFixed(1)}</span>
                </div>
                <div className="text-[10px] text-muted-foreground mt-1">
                  Over {(prediction.over_probability * 100).toFixed(0)}% · Under {(prediction.under_probability * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-xs font-semibold text-muted-foreground mb-1.5">Model Agreement</div>
                <Progress value={prediction.ensemble_agreement * 100} className="h-1.5" />
                <div className="text-[10px] text-muted-foreground mt-1">
                  {(prediction.ensemble_agreement * 100).toFixed(0)}% consensus across models
                </div>
              </div>
              <div>
                <div className="text-xs font-semibold text-muted-foreground mb-1.5">Expected Value</div>
                <div className="text-lg font-bold">
                  <span className={prediction.expected_value > 0 ? "text-emerald-500" : "text-red-500"}>
                    {prediction.expected_value > 0 ? "+" : ""}${prediction.expected_value.toFixed(2)}
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground">per $100 wagered</div>
              </div>
              {prediction.line_edge_signal && (
                <div className="sm:col-span-3 border-t pt-3 mt-1">
                  <div className="text-xs font-semibold text-muted-foreground mb-1.5">Line Edge Analysis</div>
                  <div className="flex items-center gap-4 text-xs">
                    <div>
                      <span className="text-muted-foreground">L10 Avg: </span>
                      <span className="font-mono font-bold">{prediction.l10_avg?.toFixed(1) ?? "—"}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">vs Line: </span>
                      <span className={`font-mono font-bold ${
                        (prediction.avg_vs_line_pct ?? 0) >= 50 ? "text-amber-500" :
                        (prediction.avg_vs_line_pct ?? 0) >= 30 ? "text-green-500" :
                        (prediction.avg_vs_line_pct ?? 0) <= -50 ? "text-red-500" : "text-muted-foreground"
                      }`}>
                        {(prediction.avg_vs_line_pct ?? 0) > 0 ? "+" : ""}{prediction.avg_vs_line_pct?.toFixed(1) ?? "—"}%
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Games Over: </span>
                      <span className="font-mono font-bold">{prediction.pct_games_over_line != null ? `${(prediction.pct_games_over_line * 100).toFixed(0)}%` : "—"}</span>
                    </div>
                    {prediction.line_edge_signal === "strong_over" && ["threes", "points"].includes(prediction.prop_type) && (
                      <Badge className="text-[9px] bg-amber-500/10 text-amber-600 border-amber-500/30">
                        Backtested: +22% ROI over 299 bets
                      </Badge>
                    )}
                    {prediction.line_edge_signal === "strong_under" && (
                      <Badge className="text-[9px] bg-violet-500/10 text-violet-600 border-violet-500/30">
                        Backtested: +26% ROI (under, plus-odds)
                      </Badge>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function Home() {
  const [selectedGame, setSelectedGame] = useState<string | null>(null);
  const [propFilter, setPropFilter] = useState("all");
  const [confidenceFilter, setConfidenceFilter] = useState("all");
  const [sortBy, setSortBy] = useState<"confidence" | "edge" | "ev" | "line_edge">("confidence");

  const filters = useMemo(
    () => ({
      prop_type: propFilter === "all" ? undefined : propFilter,
      min_confidence: confidenceFilter === "all" ? undefined : parseFloat(confidenceFilter),
    }),
    [propFilter, confidenceFilter]
  );

  const { data, isLoading, refetch, isFetching } = useTodayPredictions(filters);
  const generateMutation = useGeneratePredictions();
  const { data: newsData } = useNewsSentiment();
  const predictions = useMemo(() => data?.predictions ?? [], [data]);
  const games = useMemo(() => data?.games ?? [], [data]);
  const playerSentiment = useMemo(() => newsData?.player_sentiment ?? {}, [newsData]);

  // Auto-select first game when data loads
  const activeGame = selectedGame ?? (games.length > 0 ? games[0].game_id : null);

  // Filter predictions by selected game
  const gamePredictions = useMemo(() => {
    if (!activeGame) return predictions;
    return predictions.filter((p) => p.game_id === activeGame);
  }, [predictions, activeGame]);

  const sorted = useMemo(() => {
    const copy = [...gamePredictions];
    if (sortBy === "confidence") copy.sort((a, b) => b.confidence_score - a.confidence_score);
    else if (sortBy === "edge") copy.sort((a, b) => b.edge_pct - a.edge_pct);
    else if (sortBy === "ev") copy.sort((a, b) => b.expected_value - a.expected_value);
    else if (sortBy === "line_edge") {
      // Sort by line edge signal strength: strong_over first, then by avg_vs_line_pct
      const signalOrder: Record<string, number> = { strong_over: 0, moderate_over: 1, strong_under: 2, moderate_under: 3 };
      copy.sort((a, b) => {
        const aOrder = a.line_edge_signal ? (signalOrder[a.line_edge_signal] ?? 99) : 99;
        const bOrder = b.line_edge_signal ? (signalOrder[b.line_edge_signal] ?? 99) : 99;
        if (aOrder !== bOrder) return aOrder - bOrder;
        return (b.avg_vs_line_pct ?? 0) - (a.avg_vs_line_pct ?? 0);
      });
    }
    return copy;
  }, [gamePredictions, sortBy]);

  // Stats for selected game only
  const highConfidence = gamePredictions.filter((p) => p.confidence_tier >= 4).length;
  const edgeSignals = gamePredictions.filter((p) =>
    (p.line_edge_signal === "strong_over" && ["threes", "points"].includes(p.prop_type)) ||
    p.line_edge_signal === "strong_under"
  ).length;
  const avgEdge = gamePredictions.length > 0
    ? gamePredictions.reduce((s, p) => s + p.edge_pct, 0) / gamePredictions.length
    : 0;
  const totalWager = gamePredictions.reduce((s, p) => s + p.kelly_bet_size, 0);

  const activeGameInfo = games.find((g) => g.game_id === activeGame);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Today&apos;s Picks</h1>
          <p className="text-muted-foreground text-sm">AI-powered NBA player prop predictions</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
            <RefreshCw className={`mr-2 h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button size="sm" onClick={() => generateMutation.mutate()} disabled={generateMutation.isPending}>
            <Zap className="mr-2 h-4 w-4" />
            {generateMutation.isPending ? "Generating..." : "Generate Picks"}
          </Button>
        </div>
      </div>

      {/* Game Selector */}
      {games.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-muted-foreground mb-3 uppercase tracking-wider">
            Select a Game · {games.length} {games.length === 1 ? "game" : "games"} today
          </h2>
          <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            {games.map((game) => (
              <GameCard
                key={game.game_id}
                game={game}
                isSelected={game.game_id === activeGame}
                onClick={() => setSelectedGame(game.game_id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Summary Cards — scoped to selected game */}
      {activeGameInfo && (
        <>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold">
              {activeGameInfo.away_team} @ {activeGameInfo.home_team}
            </h2>
            <span className="text-sm text-muted-foreground">
              {activeGameInfo.away_team_name} @ {activeGameInfo.home_team_name}
            </span>
          </div>

          <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Picks</CardTitle>
                <Target className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{gamePredictions.length}</div>
                <p className="text-xs text-muted-foreground">
                  {new Set(gamePredictions.map((p) => p.player_name)).size} players
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">High Confidence</CardTitle>
                <TrendingUp className="h-4 w-4 text-emerald-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-emerald-500">{highConfidence}</div>
                <p className="text-xs text-muted-foreground">4-5 star picks</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avg Edge</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">+{avgEdge.toFixed(1)}%</div>
                <p className="text-xs text-muted-foreground">Over sportsbook lines</p>
              </CardContent>
            </Card>
            <Card className={edgeSignals > 0 ? "ring-1 ring-amber-500/30 bg-amber-500/5" : ""}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Line Edge Picks</CardTitle>
                <Zap className="h-4 w-4 text-amber-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-amber-500">{edgeSignals}</div>
                <p className="text-xs text-muted-foreground">Backtested +22% ROI</p>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2">
        <Filter className="h-4 w-4 text-muted-foreground" />
        <Select value={propFilter} onValueChange={setPropFilter}>
          <SelectTrigger className="w-[130px] h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PROP_TYPES.map((pt) => (
              <SelectItem key={pt.value} value={pt.value}>{pt.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={confidenceFilter} onValueChange={setConfidenceFilter}>
          <SelectTrigger className="w-[140px] h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CONFIDENCE_FILTERS.map((cf) => (
              <SelectItem key={cf.value} value={cf.value}>{cf.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={sortBy} onValueChange={(v) => setSortBy(v as "confidence" | "edge" | "ev" | "line_edge")}>
          <SelectTrigger className="w-[140px] h-8 text-xs">
            <ArrowUpDown className="h-3 w-3 mr-1" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="confidence">By Confidence</SelectItem>
            <SelectItem value="edge">By Edge %</SelectItem>
            <SelectItem value="ev">By EV</SelectItem>
            <SelectItem value="line_edge">By Line Edge</SelectItem>
          </SelectContent>
        </Select>
        {gamePredictions.length > 0 && (
          <span className="text-xs text-muted-foreground ml-auto">
            Showing {sorted.length} of {gamePredictions.length} picks
          </span>
        )}
      </div>

      {/* Picks List */}
      {isLoading ? (
        <div className="flex h-[300px] items-center justify-center">
          <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : games.length === 0 ? (
        <Card>
          <CardContent className="flex h-[300px] items-center justify-center">
            <div className="text-center">
              <Target className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
              <p className="text-lg font-medium text-muted-foreground">No predictions available</p>
              <p className="text-sm text-muted-foreground mt-1">Click &quot;Generate Picks&quot; to run the prediction engine</p>
            </div>
          </CardContent>
        </Card>
      ) : sorted.length === 0 ? (
        <Card>
          <CardContent className="flex h-[200px] items-center justify-center">
            <p className="text-sm text-muted-foreground">No picks match your filters for this game</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {sorted.map((prediction) => (
            <PredictionCard
              key={prediction.id}
              prediction={prediction}
              sentiment={playerSentiment[prediction.player_name.toLowerCase()]}
            />
          ))}
        </div>
      )}
    </div>
  );
}
