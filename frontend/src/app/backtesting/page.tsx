"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  Play,
  BarChart3,
  Target,
  TrendingUp,
  TrendingDown,
  DollarSign,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Flame,
} from "lucide-react";
import {
  useRunBacktest,
  type BacktestResult,
  type BacktestBet,
} from "@/lib/hooks/use-predictions";

const PROP_LABELS: Record<string, string> = {
  points: "PTS",
  rebounds: "REB",
  assists: "AST",
  threes: "3PM",
  steals: "STL",
  blocks: "BLK",
  turnovers: "TO",
};

const PRESET_CONFIG: Record<string, { kelly: number; confidence: number }> = {
  conservative: { kelly: 0.25, confidence: 75 },
  balanced: { kelly: 0.5, confidence: 60 },
  aggressive: { kelly: 1.0, confidence: 40 },
};

export default function BacktestingPage() {
  const [period, setPeriod] = useState("month");
  const [preset, setPreset] = useState("balanced");
  const [bankroll, setBankroll] = useState("1000");
  const [propFilter, setPropFilter] = useState("all");
  const [simulatedProgress, setSimulatedProgress] = useState(0);

  const backtestMutation = useRunBacktest();
  const result: BacktestResult | undefined = backtestMutation.data;
  const isRunning = backtestMutation.isPending;
  const hasResult = result?.status === "completed";
  const hasError = result?.status === "error";

  const handleRun = () => {
    setSimulatedProgress(0);
    const presetCfg = PRESET_CONFIG[preset] || PRESET_CONFIG.balanced;

    // Simulate progress bar while waiting for response
    const interval = setInterval(() => {
      setSimulatedProgress((prev) => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + Math.random() * 8;
      });
    }, 500);

    backtestMutation.mutate(
      {
        period,
        prop_types: propFilter !== "all" ? propFilter : undefined,
        min_confidence: presetCfg.confidence,
        bankroll: parseFloat(bankroll) || 1000,
        kelly_fraction: presetCfg.kelly,
      },
      {
        onSettled: () => {
          clearInterval(interval);
          setSimulatedProgress(100);
        },
      }
    );
  };

  const summary = result?.summary;
  const equityCurve = result?.equity_curve ?? [];
  const betLog = result?.bet_log ?? [];
  const byProp = result?.by_prop_type ?? {};

  // Build prop breakdown chart data
  const propChartData = useMemo(() => {
    return Object.entries(byProp).map(([prop, data]) => ({
      prop: PROP_LABELS[prop] || prop,
      wins: data.hits,
      losses: data.misses,
      hitRate: Math.round(data.hit_rate * 100),
      profit: data.profit,
    }));
  }, [byProp]);

  // Format equity curve for chart
  const chartData = useMemo(() => {
    return equityCurve.map((pt) => ({
      date: pt.date,
      bankroll: pt.bankroll,
    }));
  }, [equityCurve]);

  const progressPct = isRunning ? Math.min(simulatedProgress, 95) : hasResult ? 100 : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Backtesting</h1>
        <p className="text-muted-foreground">
          Test model performance against historical game data with simulated betting
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
        {/* Config Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Time Period</Label>
              <Select value={period} onValueChange={setPeriod}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="day">Last Day</SelectItem>
                  <SelectItem value="week">Last Week</SelectItem>
                  <SelectItem value="month">Last Month</SelectItem>
                  <SelectItem value="season">Full 2024-25 Season</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Model Preset</Label>
              <Select value={preset} onValueChange={setPreset}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="conservative">Conservative (1/4 Kelly, 75%+ conf)</SelectItem>
                  <SelectItem value="balanced">Balanced (1/2 Kelly, 60%+ conf)</SelectItem>
                  <SelectItem value="aggressive">Aggressive (Full Kelly, 40%+ conf)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Starting Bankroll ($)</Label>
              <Input
                type="number"
                value={bankroll}
                onChange={(e) => setBankroll(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Prop Types</Label>
              <Select value={propFilter} onValueChange={setPropFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Props</SelectItem>
                  <SelectItem value="points">Points</SelectItem>
                  <SelectItem value="rebounds">Rebounds</SelectItem>
                  <SelectItem value="assists">Assists</SelectItem>
                  <SelectItem value="threes">3-Pointers</SelectItem>
                  <SelectItem value="steals">Steals</SelectItem>
                  <SelectItem value="blocks">Blocks</SelectItem>
                  <SelectItem value="turnovers">Turnovers</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button
              className="w-full"
              onClick={handleRun}
              disabled={isRunning}
            >
              {isRunning ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Running Simulation...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Backtest
                </>
              )}
            </Button>

            {(isRunning || hasResult) && (
              <div className="space-y-1">
                <Progress value={progressPct} className="h-2" />
                <p className="text-xs text-muted-foreground text-center">
                  {isRunning
                    ? `${Math.round(progressPct)}% — Simulating game days...`
                    : hasResult
                    ? "100% — Complete!"
                    : ""}
                </p>
              </div>
            )}

            {hasResult && summary && (
              <div className="rounded-lg bg-muted/50 p-3 space-y-1 text-xs">
                <p className="font-medium text-sm">Quick Summary</p>
                <p>{summary.game_dates_simulated} game days simulated</p>
                <p>{result.config?.start_date} → {result.config?.end_date}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Panel */}
        <div className="space-y-4">
          {!hasResult && !isRunning && !hasError ? (
            <Card>
              <CardContent className="flex h-[500px] items-center justify-center">
                <div className="text-center">
                  <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground mb-3" />
                  <p className="text-lg font-medium text-muted-foreground">
                    Configure and run a backtest
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Replays the model against real historical game data
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : isRunning ? (
            <Card>
              <CardContent className="flex h-[500px] items-center justify-center">
                <div className="text-center">
                  <RefreshCw className="h-12 w-12 mx-auto text-muted-foreground mb-3 animate-spin" />
                  <p className="text-lg font-medium text-muted-foreground">
                    Running simulation...
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Replaying predictions against {period === "season" ? "full season" : `last ${period}`} of game data
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : hasError ? (
            <Card>
              <CardContent className="flex h-[300px] items-center justify-center">
                <div className="text-center">
                  <XCircle className="h-12 w-12 mx-auto text-red-500 mb-3" />
                  <p className="text-lg font-medium text-red-500">Backtest Failed</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    {result?.message || "Unknown error"}
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Summary Cards */}
              <div className="grid gap-4 md:grid-cols-5">
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Record</CardTitle>
                    <Target className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {summary?.wins}-{summary?.losses}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {summary?.total_bets} total bets
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Hit Rate</CardTitle>
                    <Target className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {((summary?.hit_rate ?? 0) * 100).toFixed(1)}%
                    </div>
                    <p className="text-xs text-muted-foreground">
                      vs 52.4% breakeven
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">ROI</CardTitle>
                    {(summary?.roi ?? 0) >= 0 ? (
                      <TrendingUp className="h-4 w-4 text-green-500" />
                    ) : (
                      <TrendingDown className="h-4 w-4 text-red-500" />
                    )}
                  </CardHeader>
                  <CardContent>
                    <div className={`text-2xl font-bold ${(summary?.roi ?? 0) >= 0 ? "text-green-500" : "text-red-500"}`}>
                      {(summary?.roi ?? 0) >= 0 ? "+" : ""}{summary?.roi?.toFixed(1)}%
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Return on investment
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">P&L</CardTitle>
                    <DollarSign className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className={`text-2xl font-bold ${(summary?.total_profit ?? 0) >= 0 ? "text-green-500" : "text-red-500"}`}>
                      {(summary?.total_profit ?? 0) >= 0 ? "+" : ""}${summary?.total_profit?.toFixed(2)}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Final: ${summary?.final_bankroll?.toFixed(2)}
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
                    <Flame className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-orange-500">
                      {summary?.max_drawdown?.toFixed(1)}%
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Streak: {summary?.current_streak} {summary?.streak_type === "win" ? "W" : "L"}
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Equity Curve */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Equity Curve</CardTitle>
                </CardHeader>
                <CardContent>
                  {chartData.length > 1 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 11 }}
                          tickFormatter={(v) => {
                            const d = new Date(v + "T00:00:00");
                            return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
                          }}
                        />
                        <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `$${v}`} />
                        <Tooltip
                          formatter={(value: number | undefined) => [`$${(value ?? 0).toFixed(2)}`, "Bankroll"]}
                          labelFormatter={(label) => {
                            const d = new Date(label + "T00:00:00");
                            return d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
                          }}
                        />
                        <defs>
                          <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <Area
                          type="monotone"
                          dataKey="bankroll"
                          stroke="hsl(var(--primary))"
                          fill="url(#equityGrad)"
                          strokeWidth={2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                      Not enough data points for chart
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Prop Type Breakdown */}
              {propChartData.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Performance by Prop Type</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={propChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="prop" tick={{ fontSize: 12 }} />
                        <YAxis tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="wins" name="Wins" fill="hsl(142, 71%, 45%)" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="losses" name="Losses" fill="hsl(0, 72%, 51%)" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-7 gap-2 mt-4">
                      {propChartData.map((p) => (
                        <div key={p.prop} className="text-center p-2 rounded-lg bg-muted/50">
                          <div className="text-xs text-muted-foreground">{p.prop}</div>
                          <div className="text-sm font-bold">{p.hitRate}%</div>
                          <div className={`text-xs ${p.profit >= 0 ? "text-green-500" : "text-red-500"}`}>
                            {p.profit >= 0 ? "+" : ""}${p.profit.toFixed(0)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Recent Bet Log */}
              {betLog.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base flex items-center justify-between">
                      Bet Log
                      <Badge variant="outline" className="text-xs font-normal">
                        Showing {betLog.length} of {result?.total_bet_log_size ?? betLog.length}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="max-h-[400px] overflow-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Date</TableHead>
                            <TableHead>Player</TableHead>
                            <TableHead>Prop</TableHead>
                            <TableHead className="text-right">Line</TableHead>
                            <TableHead className="text-right">Predicted</TableHead>
                            <TableHead className="text-right">Actual</TableHead>
                            <TableHead>Bet</TableHead>
                            <TableHead className="text-right">Profit</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {betLog.slice(-50).reverse().map((bet, i) => (
                            <TableRow key={i}>
                              <TableCell className="text-xs">{bet.date}</TableCell>
                              <TableCell className="font-medium text-sm">{bet.player}</TableCell>
                              <TableCell>
                                <Badge variant="outline" className="text-[10px]">
                                  {PROP_LABELS[bet.prop_type] || bet.prop_type}
                                </Badge>
                              </TableCell>
                              <TableCell className="text-right font-mono text-sm">{bet.line}</TableCell>
                              <TableCell className="text-right font-mono text-sm">{bet.predicted}</TableCell>
                              <TableCell className="text-right font-mono text-sm font-bold">{bet.actual}</TableCell>
                              <TableCell>
                                <div className="flex items-center gap-1">
                                  {bet.hit ? (
                                    <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                                  ) : (
                                    <XCircle className="h-3.5 w-3.5 text-red-500" />
                                  )}
                                  <span className="text-xs uppercase">{bet.bet}</span>
                                </div>
                              </TableCell>
                              <TableCell className={`text-right font-mono text-sm font-bold ${bet.profit >= 0 ? "text-green-500" : "text-red-500"}`}>
                                {bet.profit >= 0 ? "+" : ""}${bet.profit.toFixed(2)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
