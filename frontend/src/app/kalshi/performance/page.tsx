"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  AreaChart,
  Area,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Target,
  BarChart3,
  DollarSign,
  Activity,
  Zap,
  Award,
} from "lucide-react";
import {
  useAgentPerformance,
  type AgentPerformance,
} from "@/lib/hooks/use-agent";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

const STRATEGY_COLORS: Record<string, string> = {
  weather: "#3b82f6",
  crypto: "#f59e0b",
  sports: "#10b981",
  finance: "#8b5cf6",
  econ: "#ef4444",
  nba_props: "#ec4899",
  arbitrage: "#06b6d4",
};

const formatCurrencyValue = (value: unknown) => {
  const normalizedValue = Array.isArray(value) ? value[0] : value;
  const numericValue =
    typeof normalizedValue === "number"
      ? normalizedValue
      : Number(normalizedValue ?? 0);
  return `$${numericValue.toFixed(2)}`;
};

function StatCard({
  label,
  value,
  icon: Icon,
  color = "text-foreground",
  subtext,
}: {
  label: string;
  value: string;
  icon: React.ElementType;
  color?: string;
  subtext?: string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-2 mb-1">
          <Icon className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">{label}</span>
        </div>
        <p className={`text-2xl font-bold ${color}`}>{value}</p>
        {subtext && (
          <p className="text-xs text-muted-foreground mt-0.5">{subtext}</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function KalshiPerformancePage() {
  const { data: perf, isLoading } = useAgentPerformance();
  const { data: clvData } = useQuery({
    queryKey: ["agent-clv"],
    queryFn: () =>
      api.get<{
        total_trades: number;
        avg_clv_cents: number;
        positive_clv_pct: number;
        by_strategy: Record<
          string,
          { count: number; avg_clv_cents: number; positive_clv_pct: number }
        >;
      }>("/api/kalshi/agent/clv"),
    refetchInterval: 60_000,
  });

  const overall = perf?.overall;
  const byStrategy = perf?.by_strategy ?? {};
  const dailyPnl = perf?.daily_pnl ?? [];

  // Equity curve
  const equityCurve = useMemo(() => {
    if (!dailyPnl.length) return [];
    let cumulative = 0;
    let maxEquity = 0;
    return dailyPnl
      .slice()
      .reverse()
      .map((d) => {
        cumulative += d.net_pnl;
        maxEquity = Math.max(maxEquity, cumulative);
        const drawdown = cumulative - maxEquity;
        return {
          date: d.date,
          equity: Math.round(cumulative * 100) / 100,
          drawdown: Math.round(drawdown * 100) / 100,
        };
      });
  }, [dailyPnl]);

  // Strategy pie data
  const stratPieData = useMemo(() => {
    return Object.entries(byStrategy).map(([name, stats]) => ({
      name,
      trades: (stats as any).total_trades || 0,
      pnl: (stats as any).total_pnl || 0,
      color: STRATEGY_COLORS[name] || "#6b7280",
    }));
  }, [byStrategy]);

  // Strategy bar data
  const stratBarData = useMemo(() => {
    return Object.entries(byStrategy).map(([name, stats]) => {
      const s = stats as any;
      const winRate = s.total_trades > 0 ? ((s.wins || 0) / s.total_trades) * 100 : 0;
      return {
        name,
        pnl: Math.round((s.total_pnl || 0) * 100) / 100,
        trades: s.total_trades || 0,
        winRate: Math.round(winRate),
        wins: s.wins || 0,
        losses: s.losses || 0,
      };
    });
  }, [byStrategy]);

  // Daily P&L bar chart
  const dailyBarData = useMemo(() => {
    return dailyPnl
      .slice()
      .reverse()
      .slice(-30)
      .map((d) => ({
        date: d.date.slice(5), // MM-DD
        pnl: Math.round(d.net_pnl * 100) / 100,
        trades: d.trades || 0,
      }));
  }, [dailyPnl]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Activity className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const totalPnl = overall?.total_pnl ?? 0;
  const winRate = overall?.win_rate ?? 0;
  const roi = overall?.roi ?? 0;
  const totalTrades = overall?.total_trades ?? 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">
          Agent Performance
        </h1>
        <p className="text-sm text-muted-foreground">
          Kalshi autonomous trading agent analytics
        </p>
      </div>

      {/* Top KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Total P&L"
          value={`$${totalPnl >= 0 ? "+" : ""}${totalPnl.toFixed(2)}`}
          icon={DollarSign}
          color={totalPnl >= 0 ? "text-emerald-600" : "text-red-500"}
          subtext={`${totalTrades} trades`}
        />
        <StatCard
          label="Win Rate"
          value={`${(winRate * 100).toFixed(1)}%`}
          icon={Target}
          color={winRate >= 0.5 ? "text-emerald-600" : "text-amber-500"}
          subtext={`${overall?.wins ?? 0}W / ${overall?.losses ?? 0}L`}
        />
        <StatCard
          label="ROI"
          value={`${(roi * 100).toFixed(1)}%`}
          icon={TrendingUp}
          color={roi >= 0 ? "text-emerald-600" : "text-red-500"}
          subtext={`$${(overall?.total_wagered ?? 0).toFixed(0)} wagered`}
        />
        <StatCard
          label="Avg CLV"
          value={`${(clvData?.avg_clv_cents ?? 0) > 0 ? "+" : ""}${clvData?.avg_clv_cents ?? 0}c`}
          icon={Zap}
          color={(clvData?.avg_clv_cents ?? 0) > 0 ? "text-emerald-600" : "text-red-500"}
          subtext={`${clvData?.positive_clv_pct ?? 0}% positive CLV`}
        />
      </div>

      <Tabs defaultValue="equity" className="space-y-4">
        <TabsList>
          <TabsTrigger value="equity">Equity Curve</TabsTrigger>
          <TabsTrigger value="daily">Daily P&L</TabsTrigger>
          <TabsTrigger value="strategy">By Strategy</TabsTrigger>
          <TabsTrigger value="clv">CLV Analysis</TabsTrigger>
        </TabsList>

        {/* Equity Curve Tab */}
        <TabsContent value="equity">
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Equity Curve & Drawdown
              </CardTitle>
            </CardHeader>
            <CardContent>
              {equityCurve.length > 0 ? (
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={equityCurve}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip
                      formatter={(value, name) => [
                        formatCurrencyValue(value),
                        name === "equity" ? "Equity" : "Drawdown",
                      ]}
                    />
                    <Area
                      type="monotone"
                      dataKey="equity"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.1}
                      strokeWidth={2}
                    />
                    <Area
                      type="monotone"
                      dataKey="drawdown"
                      stroke="#ef4444"
                      fill="#ef4444"
                      fillOpacity={0.1}
                      strokeWidth={1}
                    />
                    <Legend />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-center text-muted-foreground py-12">
                  No daily P&L data yet
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Daily P&L Tab */}
        <TabsContent value="daily">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Daily P&L (Last 30 Days)</CardTitle>
            </CardHeader>
            <CardContent>
              {dailyBarData.length > 0 ? (
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={dailyBarData}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip
                      formatter={(value) => [formatCurrencyValue(value), "P&L"]}
                    />
                    <Bar
                      dataKey="pnl"
                      fill="#10b981"
                      radius={[4, 4, 0, 0]}
                    >
                      {dailyBarData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={entry.pnl >= 0 ? "#10b981" : "#ef4444"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-center text-muted-foreground py-12">
                  No daily data yet
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* By Strategy Tab */}
        <TabsContent value="strategy">
          <div className="grid md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">P&L by Strategy</CardTitle>
              </CardHeader>
              <CardContent>
                {stratBarData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={stratBarData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                      <XAxis type="number" tick={{ fontSize: 11 }} />
                      <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={70} />
                      <Tooltip
                        formatter={(value) => [formatCurrencyValue(value), "P&L"]}
                      />
                      <Bar dataKey="pnl" radius={[0, 4, 4, 0]}>
                        {stratBarData.map((entry, i) => (
                          <Cell
                            key={i}
                            fill={entry.pnl >= 0 ? "#10b981" : "#ef4444"}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-center text-muted-foreground py-8">No data</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Trade Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {stratPieData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={stratPieData}
                        dataKey="trades"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label={({ name, value }) => `${name}: ${value ?? 0}`}
                      >
                        {stratPieData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-center text-muted-foreground py-8">No data</p>
                )}
              </CardContent>
            </Card>

            {/* Strategy Detail Cards */}
            <div className="md:col-span-2 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {stratBarData.map((s) => (
                <Card key={s.name}>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div
                          className="h-3 w-3 rounded-full"
                          style={{
                            backgroundColor:
                              STRATEGY_COLORS[s.name] || "#6b7280",
                          }}
                        />
                        <span className="font-medium text-sm capitalize">
                          {s.name.replace("_", " ")}
                        </span>
                      </div>
                      <Badge
                        variant="outline"
                        className={
                          s.pnl >= 0
                            ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30"
                            : "bg-red-500/10 text-red-500 border-red-500/30"
                        }
                      >
                        ${s.pnl >= 0 ? "+" : ""}
                        {s.pnl.toFixed(2)}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                      <div>
                        <span className="block font-medium text-foreground">
                          {s.trades}
                        </span>
                        Trades
                      </div>
                      <div>
                        <span className="block font-medium text-foreground">
                          {s.winRate}%
                        </span>
                        Win Rate
                      </div>
                      <div>
                        <span className="block font-medium text-foreground">
                          {s.wins}W / {s.losses}L
                        </span>
                        Record
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        {/* CLV Analysis Tab */}
        <TabsContent value="clv">
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Award className="h-4 w-4" />
                Closing Line Value (CLV) — Edge Proof
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                CLV measures whether the market moved toward your position after
                you bought. Positive CLV = real edge. This is the #1 metric
                sharp bettors use.
              </p>
              {clvData && clvData.total_trades > 0 ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="p-3 text-center">
                        <p className="text-xs text-muted-foreground">
                          Avg CLV
                        </p>
                        <p
                          className={`text-xl font-bold ${clvData.avg_clv_cents > 0 ? "text-emerald-600" : "text-red-500"}`}
                        >
                          {clvData.avg_clv_cents > 0 ? "+" : ""}
                          {clvData.avg_clv_cents}c
                        </p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-3 text-center">
                        <p className="text-xs text-muted-foreground">
                          Positive CLV %
                        </p>
                        <p
                          className={`text-xl font-bold ${clvData.positive_clv_pct > 50 ? "text-emerald-600" : "text-amber-500"}`}
                        >
                          {clvData.positive_clv_pct}%
                        </p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-3 text-center">
                        <p className="text-xs text-muted-foreground">
                          Trades w/ CLV
                        </p>
                        <p className="text-xl font-bold">
                          {clvData.total_trades}
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  {/* CLV by Strategy */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">CLV by Strategy</h4>
                    {Object.entries(clvData.by_strategy || {}).map(
                      ([strat, data]) => (
                        <div
                          key={strat}
                          className="flex items-center justify-between p-2 rounded-lg bg-muted/50"
                        >
                          <div className="flex items-center gap-2">
                            <div
                              className="h-2.5 w-2.5 rounded-full"
                              style={{
                                backgroundColor:
                                  STRATEGY_COLORS[strat] || "#6b7280",
                              }}
                            />
                            <span className="text-sm font-medium capitalize">
                              {strat.replace("_", " ")}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              ({data.count} trades)
                            </span>
                          </div>
                          <div className="flex items-center gap-3">
                            <span
                              className={`text-sm font-medium ${data.avg_clv_cents > 0 ? "text-emerald-600" : "text-red-500"}`}
                            >
                              {data.avg_clv_cents > 0 ? "+" : ""}
                              {data.avg_clv_cents}c avg
                            </span>
                            <Badge variant="outline" className="text-xs">
                              {data.positive_clv_pct}% positive
                            </Badge>
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>
              ) : (
                <p className="text-center text-muted-foreground py-12">
                  No CLV data yet — CLV is recorded when trades settle
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
