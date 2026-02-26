"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Bot,
  Play,
  Square,
  ShieldAlert,
  ShieldCheck,
  CloudSun,
  Trophy,
  TrendingUp,
  TrendingDown,
  Activity,
  DollarSign,
  BarChart3,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
  Bitcoin,
  LineChart,
  Landmark,
  Dribbble,
  ChevronDown,
  ChevronRight,
  FileText,
} from "lucide-react";
import {
  useAgentStatus,
  useAgentPerformance,
  useAgentTrades,
  useAgentSignals,
  useAgentLog,
  useAgentPositions,
  useStartAgent,
  useStopAgent,
  useSetKillSwitch,
  useSetPaperMode,
  useToggleStrategy,
  useRunWeatherCycle,
  useRunSportsCycle,
  useRunCryptoCycle,
  useRunFinanceCycle,
  useRunEconCycle,
  useRunNbaPropsCycle,
  type AgentTrade,
} from "@/lib/hooks/use-agent";

function StatusBadge({ active, label }: { active: boolean; label: string }) {
  return (
    <Badge
      variant="outline"
      className={
        active
          ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30"
          : "bg-gray-500/10 text-gray-500 border-gray-500/30"
      }
    >
      <span
        className={`mr-1.5 h-2 w-2 rounded-full inline-block ${
          active ? "bg-emerald-500 animate-pulse" : "bg-gray-400"
        }`}
      />
      {label}
    </Badge>
  );
}

function PnlDisplay({ value }: { value: number }) {
  const color =
    value > 0
      ? "text-emerald-600"
      : value < 0
      ? "text-red-500"
      : "text-muted-foreground";
  return (
    <span className={`font-mono font-bold ${color}`}>
      {value >= 0 ? "+" : ""}${value.toFixed(2)}
    </span>
  );
}

function TradeRow({ trade }: { trade: AgentTrade }) {
  const [expanded, setExpanded] = useState(false);
  const hasThesis = Boolean(trade.thesis);
  const isSettled = trade.action === "sell" || trade.status === "settled";

  return (
    <>
      <tr
        className={`border-b border-muted/50 hover:bg-muted/30 ${hasThesis ? "cursor-pointer" : ""}`}
        onClick={() => hasThesis && setExpanded((e) => !e)}
      >
        <td className="py-2 pr-1 w-4">
          {hasThesis ? (
            expanded ? (
              <ChevronDown className="h-3 w-3 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
            )
          ) : null}
        </td>
        <td className="py-2 pr-3 font-mono text-xs">
          {new Date(trade.timestamp).toLocaleString([], {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </td>
        <td className="py-2 pr-3">
          <Badge variant="outline" className="text-[10px]">
            {trade.strategy}
          </Badge>
        </td>
        <td
          className="py-2 pr-3 max-w-[300px] truncate"
          title={trade.market_title}
        >
          {trade.market_title || trade.ticker}
        </td>
        <td className="py-2 pr-3">
          <Badge
            variant="outline"
            className={
              trade.side === "yes"
                ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30"
                : "bg-red-500/10 text-red-600 border-red-500/30"
            }
          >
            {trade.side.toUpperCase()}
          </Badge>
        </td>
        <td className="py-2 pr-3">
          <Badge
            variant="outline"
            className={
              trade.action === "sell"
                ? "bg-orange-500/10 text-orange-600 border-orange-500/30"
                : "bg-blue-500/10 text-blue-600 border-blue-500/30"
            }
          >
            {(trade.action || "buy").toUpperCase()}
          </Badge>
        </td>
        <td className="py-2 pr-3 text-right font-mono">{trade.count}x</td>
        <td className="py-2 pr-3 text-right font-mono">{trade.price_cents}c</td>
        <td className="py-2 pr-3 text-right font-mono">
          {(trade.edge * 100).toFixed(1)}%
        </td>
        <td className="py-2 pr-3 text-right">
          {trade.pnl != null && trade.pnl !== 0 ? (
            <PnlDisplay value={trade.pnl} />
          ) : isSettled ? (
            <PnlDisplay value={trade.pnl ?? 0} />
          ) : (
            <span className="text-muted-foreground">—</span>
          )}
        </td>
        <td className="py-2">
          {isSettled ? (
            (trade.pnl ?? 0) > 0 ? (
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            ) : (
              <XCircle className="h-4 w-4 text-red-500" />
            )
          ) : (
            <Clock className="h-4 w-4 text-amber-500" />
          )}
        </td>
      </tr>
      {expanded && hasThesis && (
        <tr className="border-b border-muted/50 bg-muted/20">
          <td colSpan={11} className="px-4 py-3">
            <div className="flex gap-2 items-start">
              <FileText className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
              <div className="space-y-1">
                <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide">
                  Trade Thesis
                </p>
                <p className="text-sm text-foreground leading-relaxed">
                  {trade.thesis}
                </p>
                {trade.result && isSettled && (
                  <p className="text-xs text-muted-foreground mt-1">
                    <span className="font-medium">Outcome:</span>{" "}
                    Market resolved{" "}
                    <span
                      className={
                        (trade.pnl ?? 0) > 0
                          ? "text-emerald-600 font-semibold"
                          : "text-red-500 font-semibold"
                      }
                    >
                      {(trade.pnl ?? 0) > 0 ? "WIN" : "LOSS"}
                    </span>{" "}
                    · P&L: <PnlDisplay value={trade.pnl ?? 0} />
                  </p>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function AgentPage() {
  const [activeTab, setActiveTab] = useState<
    "positions" | "overview" | "trades" | "signals" | "quality" | "log"
  >("positions");

  const { data: status, isLoading: statusLoading } = useAgentStatus();
  const { data: performance } = useAgentPerformance();
  const { data: tradesData } = useAgentTrades({ limit: 50 });
  const { data: signalsData } = useAgentSignals({ limit: 50 });
  const { data: logData } = useAgentLog({ limit: 100 });
  const { data: positionsData } = useAgentPositions();

  const startAgent = useStartAgent();
  const stopAgent = useStopAgent();
  const setKillSwitch = useSetKillSwitch();
  const setPaperMode = useSetPaperMode();
  const toggleStrategy = useToggleStrategy();
  const runWeather = useRunWeatherCycle();
  const runSports = useRunSportsCycle();
  const runCrypto = useRunCryptoCycle();
  const runFinance = useRunFinanceCycle();
  const runEcon = useRunEconCycle();
  const runNbaProps = useRunNbaPropsCycle();

  if (statusLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const isRunning = status?.running ?? false;
  const isPaper = status?.paper_mode ?? true;
  const isKilled = status?.kill_switch ?? false;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-violet-600 text-white">
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Trading Agent</h1>
            <p className="text-sm text-muted-foreground">
              Autonomous Kalshi trading — weather & cross-market sports
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge active={isRunning} label={isRunning ? "Running" : "Stopped"} />
          <StatusBadge active={isPaper} label={isPaper ? "Paper" : "LIVE"} />
          {isKilled && (
            <Badge variant="destructive" className="gap-1">
              <ShieldAlert className="h-3 w-3" />
              Kill Switch
            </Badge>
          )}
        </div>
      </div>

      {/* Control Panel */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-4">
            {/* Start/Stop */}
            <Button
              onClick={() =>
                isRunning ? stopAgent.mutate() : startAgent.mutate()
              }
              variant={isRunning ? "destructive" : "default"}
              size="sm"
              disabled={isKilled && !isRunning}
            >
              {isRunning ? (
                <>
                  <Square className="h-4 w-4 mr-1" /> Stop Agent
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-1" /> Start Agent
                </>
              )}
            </Button>

            {/* Kill Switch */}
            <div className="flex items-center gap-2 border-l pl-4">
              <ShieldAlert
                className={`h-4 w-4 ${
                  isKilled ? "text-red-500" : "text-muted-foreground"
                }`}
              />
              <span className="text-sm">Kill Switch</span>
              <Switch
                checked={isKilled}
                onCheckedChange={(v) => setKillSwitch.mutate(v)}
              />
            </div>

            {/* Paper Mode */}
            <div className="flex items-center gap-2 border-l pl-4">
              <Activity
                className={`h-4 w-4 ${
                  isPaper ? "text-amber-500" : "text-emerald-500"
                }`}
              />
              <span className="text-sm">Paper Mode</span>
              <Switch
                checked={isPaper}
                onCheckedChange={(v) => setPaperMode.mutate(v)}
              />
            </div>

            {/* Strategy Toggles */}
            <div className="flex items-center gap-2 border-l pl-4">
              <CloudSun className="h-4 w-4 text-sky-500" />
              <span className="text-sm">Weather</span>
              <Switch
                checked={status?.strategy_enabled?.weather ?? true}
                onCheckedChange={(v) =>
                  toggleStrategy.mutate({ strategy: "weather", enabled: v })
                }
              />
            </div>

            <div className="flex items-center gap-2">
              <Trophy className="h-4 w-4 text-orange-500" />
              <span className="text-sm">Sports</span>
              <Switch
                checked={status?.strategy_enabled?.sports ?? true}
                onCheckedChange={(v) =>
                  toggleStrategy.mutate({ strategy: "sports", enabled: v })
                }
              />
            </div>

            <div className="flex items-center gap-2">
              <Bitcoin className="h-4 w-4 text-yellow-500" />
              <span className="text-sm">Crypto</span>
              <Switch
                checked={status?.strategy_enabled?.crypto ?? true}
                onCheckedChange={(v) =>
                  toggleStrategy.mutate({ strategy: "crypto", enabled: v })
                }
              />
            </div>

            <div className="flex items-center gap-2">
              <LineChart className="h-4 w-4 text-blue-500" />
              <span className="text-sm">Finance</span>
              <Switch
                checked={status?.strategy_enabled?.finance ?? true}
                onCheckedChange={(v) =>
                  toggleStrategy.mutate({ strategy: "finance", enabled: v })
                }
              />
            </div>

            <div className="flex items-center gap-2">
              <Landmark className="h-4 w-4 text-emerald-500" />
              <span className="text-sm">Econ</span>
              <Switch
                checked={status?.strategy_enabled?.econ ?? true}
                onCheckedChange={(v) =>
                  toggleStrategy.mutate({ strategy: "econ", enabled: v })
                }
              />
            </div>

            <div className="flex items-center gap-2">
              <Dribbble className="h-4 w-4 text-orange-600" />
              <span className="text-sm">NBA Props</span>
              <Switch
                checked={status?.strategy_enabled?.nba_props ?? true}
                onCheckedChange={(v) =>
                  toggleStrategy.mutate({ strategy: "nba_props", enabled: v })
                }
              />
            </div>

            {/* Manual Triggers */}
            <div className="flex flex-wrap items-center gap-2 border-l pl-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => runWeather.mutate()}
                disabled={runWeather.isPending}
              >
                <CloudSun className="h-3.5 w-3.5 mr-1" />
                {runWeather.isPending ? "Running..." : "Weather"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => runSports.mutate()}
                disabled={runSports.isPending}
              >
                <Trophy className="h-3.5 w-3.5 mr-1" />
                {runSports.isPending ? "Running..." : "Sports"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => runCrypto.mutate()}
                disabled={runCrypto.isPending}
              >
                <Bitcoin className="h-3.5 w-3.5 mr-1" />
                {runCrypto.isPending ? "Running..." : "Crypto"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => runFinance.mutate()}
                disabled={runFinance.isPending}
              >
                <LineChart className="h-3.5 w-3.5 mr-1" />
                {runFinance.isPending ? "Running..." : "Finance"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => runEcon.mutate()}
                disabled={runEcon.isPending}
              >
                <Landmark className="h-3.5 w-3.5 mr-1" />
                {runEcon.isPending ? "Running..." : "Econ"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => runNbaProps.mutate()}
                disabled={runNbaProps.isPending}
              >
                <Dribbble className="h-3.5 w-3.5 mr-1" />
                {runNbaProps.isPending ? "Running..." : "NBA Props"}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <Card className={(status?.over_deployed) ? "border-red-500" : ""}>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
              <DollarSign className="h-3.5 w-3.5" />
              Bankroll
            </div>
            <div className="text-2xl font-bold">
              ${(status?.effective_bankroll ?? status?.bankroll ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            {status?.bankroll && status?.effective_bankroll !== undefined && status.effective_bankroll !== status.bankroll && (
              <div className="text-xs text-muted-foreground mt-0.5">started ${status.bankroll.toLocaleString()}</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
              <TrendingDown className="h-3.5 w-3.5" />
              Deployed
            </div>
            <div className={`text-2xl font-bold ${(status?.over_deployed) ? "text-red-500" : "text-amber-600"}`}>
              ${(status?.total_exposure ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            {status?.over_deployed && (
              <div className="text-xs text-red-500 mt-0.5 font-semibold">⚠ Over-deployed</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
              <ShieldCheck className="h-3.5 w-3.5" />
              Remaining
            </div>
            <div className={`text-2xl font-bold ${(status?.remaining_capital ?? 0) < 0 ? "text-red-500" : "text-emerald-600"}`}>
              ${(status?.remaining_capital ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
              <BarChart3 className="h-3.5 w-3.5" />
              Today&apos;s P&L
            </div>
            <div className="text-2xl font-bold">
              <PnlDisplay value={status?.today_pnl ?? 0} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
              <Zap className="h-3.5 w-3.5" />
              Today&apos;s Trades
            </div>
            <div className="text-2xl font-bold">
              {status?.today_trades ?? 0}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
              <TrendingUp className="h-3.5 w-3.5" />
              Total P&L
            </div>
            <div className="text-2xl font-bold">
              <PnlDisplay value={performance?.overall?.total_pnl ?? 0} />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance by Strategy */}
      {performance?.by_strategy &&
        Object.keys(performance.by_strategy).length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(performance.by_strategy).map(([name, stats]) => (
              <Card key={name}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    {name === "weather" ? (
                      <CloudSun className="h-4 w-4 text-sky-500" />
                    ) : (
                      <Trophy className="h-4 w-4 text-orange-500" />
                    )}
                    {name.charAt(0).toUpperCase() + name.slice(1)} Strategy
                  </CardTitle>
                </CardHeader>
                <CardContent className="pb-4">
                  <div className="grid grid-cols-4 gap-3 text-sm">
                    <div>
                      <div className="text-muted-foreground text-xs">
                        Trades
                      </div>
                      <div className="font-semibold">{stats.total_trades}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground text-xs">
                        Win Rate
                      </div>
                      <div className="font-semibold">
                        {stats.total_trades > 0
                          ? (
                              ((stats.wins ?? 0) / stats.total_trades) *
                              100
                            ).toFixed(1)
                          : 0}
                        %
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground text-xs">P&L</div>
                      <div className="font-semibold">
                        <PnlDisplay value={stats.total_pnl ?? 0} />
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground text-xs">Fees</div>
                      <div className="font-semibold text-muted-foreground">
                        ${(stats.total_fees ?? 0).toFixed(2)}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

      {/* Tabs */}
      <div className="flex gap-1 border-b">
        {(
          [
            { key: "positions", label: "Open Positions" },
            { key: "overview", label: "Daily P&L" },
            { key: "trades", label: "Trades" },
            { key: "signals", label: "Signals" },
            { key: "quality", label: "Signal Quality" },
            { key: "log", label: "Agent Log" },
          ] as const
        ).map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.key
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "positions" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center justify-between">
              <span>Open Positions ({positionsData?.total ?? 0})</span>
              {positionsData && (
                <div className="flex items-center gap-4 text-sm font-normal">
                  <span className="text-muted-foreground">
                    Risked: <span className="font-mono font-semibold text-foreground">${positionsData.total_cost.toFixed(2)}</span>
                  </span>
                  <span className="text-muted-foreground">
                    Max Risk: <span className="font-mono font-semibold text-foreground">${positionsData.total_max_risk.toFixed(2)}</span>
                  </span>
                  <span className="text-muted-foreground">
                    Unrealized: <PnlDisplay value={positionsData.total_unrealized_pnl} />
                  </span>
                </div>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {positionsData?.positions && positionsData.positions.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-2 pr-3">Market</th>
                      <th className="pb-2 pr-3">Side</th>
                      <th className="pb-2 pr-3">Strategy</th>
                      <th className="pb-2 pr-3 text-right">Contracts</th>
                      <th className="pb-2 pr-3 text-right">Entry</th>
                      <th className="pb-2 pr-3 text-right">Current</th>
                      <th className="pb-2 pr-3 text-right">Cost</th>
                      <th className="pb-2 pr-3 text-right">Max Risk</th>
                      <th className="pb-2 pr-3 text-right">Unreal. P&L</th>
                      <th className="pb-2 pr-3 text-right">Edge</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positionsData.positions.map((pos) => {
                      const currentEdge = pos.current_edge;
                      const edgeColor = currentEdge === null
                        ? "text-muted-foreground"
                        : currentEdge > 0.03
                        ? "text-emerald-600"
                        : currentEdge < -0.02
                        ? "text-red-500"
                        : "text-amber-500";
                      return (
                        <tr
                          key={`${pos.ticker}-${pos.side}`}
                          className="border-b border-muted/50 hover:bg-muted/30"
                        >
                          <td
                            className="py-2 pr-3 max-w-[220px] truncate"
                            title={pos.title}
                          >
                            {pos.title || pos.ticker}
                          </td>
                          <td className="py-2 pr-3">
                            <Badge
                              variant="outline"
                              className={
                                pos.side === "yes"
                                  ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30"
                                  : "bg-red-500/10 text-red-600 border-red-500/30"
                              }
                            >
                              {pos.side.toUpperCase()}
                            </Badge>
                          </td>
                          <td className="py-2 pr-3">
                            <Badge variant="outline" className="text-[10px]">
                              {pos.strategy}
                            </Badge>
                          </td>
                          <td className="py-2 pr-3 text-right font-mono">
                            {pos.contracts}x
                          </td>
                          <td className="py-2 pr-3 text-right font-mono">
                            {pos.avg_entry_cents}c
                          </td>
                          <td className="py-2 pr-3 text-right font-mono">
                            {pos.mark_price_cents !== null
                              ? `${pos.mark_price_cents}c`
                              : "—"}
                          </td>
                          <td className="py-2 pr-3 text-right font-mono">
                            ${pos.total_cost.toFixed(2)}
                          </td>
                          <td className="py-2 pr-3 text-right font-mono text-muted-foreground">
                            ${pos.max_risk.toFixed(2)}
                          </td>
                          <td className="py-2 pr-3 text-right">
                            {pos.unrealized_pnl !== null ? (
                              <PnlDisplay value={pos.unrealized_pnl} />
                            ) : (
                              <span className="text-muted-foreground">—</span>
                            )}
                          </td>
                          <td className={`py-2 pr-3 text-right font-mono ${edgeColor}`}>
                            {currentEdge !== null
                              ? `${currentEdge > 0 ? "+" : ""}${(
                                  currentEdge * 100
                                ).toFixed(1)}%`
                              : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-8">
                No open positions. Run a strategy cycle to generate trades.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {activeTab === "overview" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Daily P&L (Last 30 Days)</CardTitle>
          </CardHeader>
          <CardContent>
            {performance?.daily_pnl && performance.daily_pnl.length > 0 ? (
              <div className="space-y-1">
                {performance.daily_pnl.map((day) => (
                  <div
                    key={day.date}
                    className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-muted/50"
                  >
                    <span className="text-sm font-mono">{day.date}</span>
                    <div className="flex items-center gap-4">
                      <span className="text-xs text-muted-foreground">
                        {day.trades} trades
                      </span>
                      <PnlDisplay value={day.net_pnl} />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-8">
                No trading data yet. Start the agent to begin paper trading.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {activeTab === "trades" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              Recent Trades ({tradesData?.total ?? 0})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {tradesData?.trades && tradesData.trades.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-2 pr-1 w-4"></th>
                      <th className="pb-2 pr-3">Time</th>
                      <th className="pb-2 pr-3">Strategy</th>
                      <th className="pb-2 pr-3">Market</th>
                      <th className="pb-2 pr-3">Side</th>
                      <th className="pb-2 pr-3">Action</th>
                      <th className="pb-2 pr-3 text-right">Size</th>
                      <th className="pb-2 pr-3 text-right">Price</th>
                      <th className="pb-2 pr-3 text-right">Edge</th>
                      <th className="pb-2 pr-3 text-right">P&L</th>
                      <th className="pb-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tradesData.trades.map((trade) => (
                      <TradeRow key={trade.id} trade={trade} />
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-8">
                No trades yet.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {activeTab === "signals" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              Recent Signals ({signalsData?.total ?? 0})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {signalsData?.signals && signalsData.signals.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-2 pr-3">Time</th>
                      <th className="pb-2 pr-3">Strategy</th>
                      <th className="pb-2 pr-3">Market</th>
                      <th className="pb-2 pr-3">Side</th>
                      <th className="pb-2 pr-3 text-right">Our Prob</th>
                      <th className="pb-2 pr-3 text-right">Kalshi</th>
                      <th className="pb-2 pr-3 text-right">Edge</th>
                      <th className="pb-2 pr-3 text-right">Confidence</th>
                      <th className="pb-2">Acted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {signalsData.signals.map((sig) => (
                      <tr
                        key={sig.id}
                        className="border-b border-muted/50 hover:bg-muted/30"
                      >
                        <td className="py-2 pr-3 font-mono text-xs">
                          {new Date(sig.timestamp).toLocaleString([], {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </td>
                        <td className="py-2 pr-3">
                          <Badge variant="outline" className="text-[10px]">
                            {sig.strategy}
                          </Badge>
                        </td>
                        <td
                          className="py-2 pr-3 max-w-[350px] truncate"
                          title={sig.market_title}
                        >
                          {sig.market_title || sig.ticker}
                        </td>
                        <td className="py-2 pr-3">
                          <Badge
                            variant="outline"
                            className={
                              sig.side === "yes"
                                ? "bg-emerald-500/10 text-emerald-600"
                                : "bg-red-500/10 text-red-600"
                            }
                          >
                            {sig.side.toUpperCase()}
                          </Badge>
                        </td>
                        <td className="py-2 pr-3 text-right font-mono">
                          {(sig.our_prob * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 pr-3 text-right font-mono">
                          {(sig.kalshi_prob * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 pr-3 text-right font-mono text-emerald-600">
                          +{(sig.edge * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 pr-3 text-right font-mono">
                          {(sig.confidence * 100).toFixed(0)}%
                        </td>
                        <td className="py-2">
                          {sig.acted_on ? (
                            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                          ) : (
                            <XCircle className="h-4 w-4 text-muted-foreground" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-8">
                No signals yet. Run a strategy cycle to generate signals.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {activeTab === "quality" && (
        <div className="space-y-4">
          {/* Overall signal quality summary */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-violet-500" />
                Win Rate by Price Bucket
                <span className="text-xs font-normal text-muted-foreground ml-1">(settled trades only)</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {performance?.by_price_bucket && performance.by_price_bucket.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left text-muted-foreground">
                        <th className="pb-2 pr-3">Price</th>
                        <th className="pb-2 pr-3">Strategy</th>
                        <th className="pb-2 pr-3">Side</th>
                        <th className="pb-2 pr-3 text-right">Trades</th>
                        <th className="pb-2 pr-3 text-right">Wins</th>
                        <th className="pb-2 pr-3 text-right">Win Rate</th>
                        <th className="pb-2 pr-3 text-right">P&L</th>
                        <th className="pb-2">Health</th>
                      </tr>
                    </thead>
                    <tbody>
                      {performance.by_price_bucket.map((row, i) => {
                        const wr = row.trades > 0 ? row.wins / row.trades : 0;
                        const wrPct = (wr * 100).toFixed(0);
                        const isGood = wr >= 0.55;
                        const isBad = wr < 0.35 && row.trades >= 3;
                        const wrColor = isGood
                          ? "text-emerald-600 font-semibold"
                          : isBad
                          ? "text-red-500 font-semibold"
                          : "text-amber-600";
                        return (
                          <tr key={i} className="border-b border-muted/50 hover:bg-muted/30">
                            <td className="py-2 pr-3 font-mono text-xs">{row.bucket}</td>
                            <td className="py-2 pr-3">
                              <Badge variant="outline" className="text-[10px]">{row.strategy}</Badge>
                            </td>
                            <td className="py-2 pr-3">
                              <Badge
                                variant="outline"
                                className={row.side === "yes"
                                  ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30 text-[10px]"
                                  : "bg-red-500/10 text-red-600 border-red-500/30 text-[10px]"}
                              >
                                {row.side.toUpperCase()}
                              </Badge>
                            </td>
                            <td className="py-2 pr-3 text-right font-mono">{row.trades}</td>
                            <td className="py-2 pr-3 text-right font-mono">{row.wins}</td>
                            <td className={`py-2 pr-3 text-right font-mono ${wrColor}`}>{wrPct}%</td>
                            <td className="py-2 pr-3 text-right">
                              <PnlDisplay value={row.total_pnl} />
                            </td>
                            <td className="py-2">
                              {row.trades < 3 ? (
                                <span className="text-xs text-muted-foreground">low n</span>
                              ) : isGood ? (
                                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                              ) : isBad ? (
                                <XCircle className="h-4 w-4 text-red-500" />
                              ) : (
                                <Clock className="h-4 w-4 text-amber-500" />
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No settled trades yet. Data will appear after markets resolve.
                </p>
              )}
            </CardContent>
          </Card>

          {/* Current thresholds */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Zap className="h-4 w-4 text-amber-500" />
                Active Filters (NBA Props)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="rounded-lg bg-muted/50 p-3">
                  <div className="text-xs text-muted-foreground mb-1">YES min price</div>
                  <div className="font-mono font-bold text-emerald-600">20c</div>
                </div>
                <div className="rounded-lg bg-muted/50 p-3">
                  <div className="text-xs text-muted-foreground mb-1">NO max price</div>
                  <div className="font-mono font-bold text-red-500">80c</div>
                </div>
                <div className="rounded-lg bg-muted/50 p-3">
                  <div className="text-xs text-muted-foreground mb-1">Min confidence</div>
                  <div className="font-mono font-bold">45%</div>
                </div>
                <div className="rounded-lg bg-muted/50 p-3">
                  <div className="text-xs text-muted-foreground mb-1">Min edge</div>
                  <div className="font-mono font-bold">8%</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === "log" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Agent Log</CardTitle>
          </CardHeader>
          <CardContent>
            {logData?.log && logData.log.length > 0 ? (
              <div className="space-y-0.5 max-h-[500px] overflow-y-auto font-mono text-xs">
                {logData.log.map((entry) => (
                  <div
                    key={entry.id}
                    className={`flex gap-2 py-1 px-2 rounded ${
                      entry.level === "error"
                        ? "bg-red-500/5 text-red-600"
                        : entry.level === "warning"
                        ? "bg-amber-500/5 text-amber-600"
                        : entry.level === "paper_trade" ||
                          entry.level === "live_trade"
                        ? "bg-emerald-500/5 text-emerald-600"
                        : "text-muted-foreground"
                    }`}
                  >
                    <span className="shrink-0 w-[140px]">
                      {new Date(entry.timestamp).toLocaleString([], {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </span>
                    <span className="shrink-0 w-[80px] uppercase">
                      [{entry.level}]
                    </span>
                    {entry.strategy && (
                      <span className="shrink-0 w-[70px] text-primary">
                        {entry.strategy}
                      </span>
                    )}
                    <span className="flex-1">{entry.message}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-8">
                No log entries yet.
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
