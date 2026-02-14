"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  DollarSign,
  BarChart3,
} from "lucide-react";
import { useBets, useBetSummary } from "@/lib/hooks/use-bets";
import { useAppSettings } from "@/lib/hooks/use-settings";

export default function BankrollPage() {
  const { data: summary } = useBetSummary();
  const { data: settings } = useAppSettings();
  const { data: betsData } = useBets({ limit: 200 });
  const bets = useMemo(() => betsData?.bets ?? [], [betsData]);

  const startingBankroll = settings?.bankroll ?? 1000;

  // Build equity curve
  const equityCurve = useMemo(() => {
    const resolved = bets
      .filter((b) => b.profit !== null && b.status !== "pending")
      .sort((a, b) => (a.created_at ?? "").localeCompare(b.created_at ?? ""));

    let balance = startingBankroll;
    const points: { bet: number; balance: number; date: string }[] = [
      { bet: 0, balance: startingBankroll, date: "Start" },
    ];
    for (let i = 0; i < resolved.length; i++) {
      balance += resolved[i].profit ?? 0;
      const dateStr = resolved[i].created_at
        ? new Date(resolved[i].created_at!).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
          })
        : `#${i + 1}`;
      points.push({
        bet: i + 1,
        balance: Math.round(balance * 100) / 100,
        date: dateStr,
      });
    }
    return points;
  }, [bets, startingBankroll]);

  const currentBalance =
    equityCurve.length > 1
      ? equityCurve[equityCurve.length - 1].balance
      : startingBankroll;
  const totalPL = currentBalance - startingBankroll;
  const plPct =
    startingBankroll > 0 ? (totalPL / startingBankroll) * 100 : 0;
  const highWaterMark = Math.max(...equityCurve.map((p) => p.balance));
  const lowPoint = Math.min(...equityCurve.map((p) => p.balance));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Bankroll</h1>
        <p className="text-muted-foreground">
          Track your bankroll over time with equity curve visualization
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Current Balance
            </CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${currentBalance.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </div>
            <p className="text-xs text-muted-foreground">
              Started at ${startingBankroll.toLocaleString()}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
            {totalPL >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-500" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-500" />
            )}
          </CardHeader>
          <CardContent>
            <div
              className={`text-2xl font-bold ${
                totalPL > 0
                  ? "text-green-500"
                  : totalPL < 0
                  ? "text-red-500"
                  : ""
              }`}
            >
              {totalPL >= 0 ? "+" : ""}${totalPL.toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground">
              {plPct >= 0 ? "+" : ""}
              {plPct.toFixed(1)}% return
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              High Water Mark
            </CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${highWaterMark.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </div>
            <p className="text-xs text-muted-foreground">Peak balance</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Unit Size</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${(settings?.unit_size ?? 10).toFixed(0)}
            </div>
            <p className="text-xs text-muted-foreground">
              {summary
                ? `${summary.total_bets} total bets`
                : "Configure in settings"}
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wallet className="h-5 w-5" />
            Equity Curve
            {equityCurve.length <= 1 && (
              <Badge variant="outline" className="text-xs">
                No data yet
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {equityCurve.length <= 1 ? (
            <div className="flex h-[350px] items-center justify-center rounded-lg border border-dashed">
              <div className="text-center">
                <Wallet className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
                <p className="text-lg font-medium text-muted-foreground">
                  No bankroll history yet
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  Log bets to see your equity curve
                </p>
              </div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={350}>
              <AreaChart data={equityCurve}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v) => `$${v}`}
                  domain={[
                    Math.floor(lowPoint * 0.95),
                    Math.ceil(highWaterMark * 1.05),
                  ]}
                />
                <Tooltip
                  formatter={(value) => [
                    `$${Number(value).toFixed(2)}`,
                    "Balance",
                  ]}
                />
                <defs>
                  <linearGradient id="eqGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor={
                        totalPL >= 0
                          ? "hsl(142, 76%, 36%)"
                          : "hsl(0, 84%, 60%)"
                      }
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="95%"
                      stopColor={
                        totalPL >= 0
                          ? "hsl(142, 76%, 36%)"
                          : "hsl(0, 84%, 60%)"
                      }
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <Area
                  type="monotone"
                  dataKey="balance"
                  stroke={
                    totalPL >= 0
                      ? "hsl(142, 76%, 36%)"
                      : "hsl(0, 84%, 60%)"
                  }
                  fill="url(#eqGradient)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
