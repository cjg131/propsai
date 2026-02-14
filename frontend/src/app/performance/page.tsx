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
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Target,
  BarChart3,
  RefreshCw,
} from "lucide-react";
import { useBets, useBetSummary } from "@/lib/hooks/use-bets";

export default function PerformancePage() {
  const { data: summary } = useBetSummary();
  const { data: betsData, isLoading } = useBets({ limit: 200 });
  const bets = useMemo(() => betsData?.bets ?? [], [betsData]);

  // Build cumulative P&L data
  const plData = useMemo(() => {
    const resolved = bets
      .filter((b) => b.profit !== null && b.status !== "pending")
      .sort((a, b) => (a.created_at ?? "").localeCompare(b.created_at ?? ""));

    let running = 0;
    const result: { bet: number; profit: number; cumulative: number; player: string; prop: string }[] = [];
    for (let i = 0; i < resolved.length; i++) {
      running += resolved[i].profit ?? 0;
      result.push({
        bet: i + 1,
        profit: resolved[i].profit ?? 0,
        cumulative: Math.round(running * 100) / 100,
        player: resolved[i].player_name,
        prop: resolved[i].prop_type,
      });
    }
    return result;
  }, [bets]);

  // Build prop type breakdown
  const propBreakdown = useMemo(() => {
    const resolved = bets.filter((b) => b.status !== "pending");
    const grouped: Record<string, { wins: number; losses: number; profit: number; total: number }> = {};
    for (const b of resolved) {
      const key = b.prop_type || "other";
      if (!grouped[key]) grouped[key] = { wins: 0, losses: 0, profit: 0, total: 0 };
      grouped[key].total++;
      if (b.status === "won") grouped[key].wins++;
      if (b.status === "lost") grouped[key].losses++;
      grouped[key].profit += b.profit ?? 0;
    }
    return Object.entries(grouped).map(([prop, data]) => ({
      prop,
      wins: data.wins,
      losses: data.losses,
      winRate: data.total > 0 ? Math.round((data.wins / data.total) * 100) : 0,
      profit: Math.round(data.profit * 100) / 100,
      total: data.total,
    }));
  }, [bets]);

  // Build drawdown data
  const drawdownData = useMemo(() => {
    let peak = 0;
    let running = 0;
    const resolved = bets
      .filter((b) => b.profit !== null && b.status !== "pending")
      .sort((a, b) => (a.created_at ?? "").localeCompare(b.created_at ?? ""));

    const result: { bet: number; drawdown: number; cumulative: number }[] = [];
    for (let i = 0; i < resolved.length; i++) {
      running += resolved[i].profit ?? 0;
      if (running > peak) peak = running;
      const drawdown = peak > 0 ? ((peak - running) / peak) * 100 : 0;
      result.push({
        bet: i + 1,
        drawdown: Math.round(drawdown * 100) / 100,
        cumulative: Math.round(running * 100) / 100,
      });
    }
    return result;
  }, [bets]);

  const emptyState = (msg: string) => (
    <div className="flex h-[350px] items-center justify-center rounded-lg border border-dashed">
      <div className="text-center">
        <BarChart3 className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
        <p className="text-sm text-muted-foreground">{msg}</p>
        <p className="text-xs text-muted-foreground mt-1">
          Log bets to see analytics
        </p>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          Performance Analytics
        </h1>
        <p className="text-muted-foreground">
          Multi-dimensional analysis of your betting performance
        </p>
      </div>

      {/* Summary Row */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {summary ? `${(summary.win_rate * 100).toFixed(1)}%` : "--%"}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ROI</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${summary && summary.roi > 0 ? "text-green-500" : summary && summary.roi < 0 ? "text-red-500" : ""}`}>
              {summary ? `${summary.roi > 0 ? "+" : ""}${summary.roi.toFixed(1)}%` : "--%"}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
            {summary && summary.total_profit >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-500" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-500" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${summary && summary.total_profit > 0 ? "text-green-500" : summary && summary.total_profit < 0 ? "text-red-500" : ""}`}>
              {summary ? `${summary.total_profit >= 0 ? "+" : ""}$${summary.total_profit.toFixed(2)}` : "$0"}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Best Prop</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize">
              {summary?.best_prop_type ?? "--"}
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">P&L Curve</TabsTrigger>
          <TabsTrigger value="by-prop">By Prop Type</TabsTrigger>
          <TabsTrigger value="drawdown">Drawdown</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <Card>
            <CardHeader>
              <CardTitle>Cumulative Profit & Loss</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex h-[350px] items-center justify-center">
                  <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : plData.length === 0 ? (
                emptyState("No resolved bets to chart")
              ) : (
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={plData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="bet" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v}`} />
                    <Tooltip
                      formatter={(value) => [`$${Number(value).toFixed(2)}`, "Cumulative P&L"]}
                      labelFormatter={(label) => `Bet #${label}`}
                    />
                    <defs>
                      <linearGradient id="plGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(142, 76%, 36%)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="hsl(142, 76%, 36%)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area
                      type="monotone"
                      dataKey="cumulative"
                      stroke="hsl(142, 76%, 36%)"
                      fill="url(#plGradient)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="by-prop">
          <Card>
            <CardHeader>
              <CardTitle>Performance by Prop Type</CardTitle>
            </CardHeader>
            <CardContent>
              {propBreakdown.length === 0 ? (
                emptyState("No prop data to display")
              ) : (
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={propBreakdown}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="prop" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="wins" name="Wins" stackId="a" fill="hsl(142, 76%, 36%)" />
                    <Bar dataKey="losses" name="Losses" stackId="a" fill="hsl(0, 84%, 60%)" />
                  </BarChart>
                </ResponsiveContainer>
              )}

              {propBreakdown.length > 0 && (
                <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {propBreakdown.map((p) => (
                    <div
                      key={p.prop}
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                    >
                      <div>
                        <span className="font-medium text-sm capitalize">{p.prop}</span>
                        <div className="text-xs text-muted-foreground">
                          {p.wins}W-{p.losses}L ({p.winRate}%)
                        </div>
                      </div>
                      <Badge
                        variant={p.profit >= 0 ? "default" : "destructive"}
                        className="font-mono"
                      >
                        {p.profit >= 0 ? "+" : ""}${p.profit.toFixed(2)}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="drawdown">
          <Card>
            <CardHeader>
              <CardTitle>Drawdown Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              {drawdownData.length === 0 ? (
                emptyState("No drawdown data to display")
              ) : (
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={drawdownData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="bet" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      formatter={(value) => [`${Number(value).toFixed(1)}%`, "Drawdown"]}
                      labelFormatter={(label) => `Bet #${label}`}
                    />
                    <defs>
                      <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area
                      type="monotone"
                      dataKey="drawdown"
                      stroke="hsl(0, 84%, 60%)"
                      fill="url(#ddGradient)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
