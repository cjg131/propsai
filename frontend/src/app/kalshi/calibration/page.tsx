"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  AlertCircle,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Activity,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from "recharts";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  useDisciplineStatus,
  useCalibrationReport,
  useMarketMakerStatus,
} from "@/lib/hooks/use-discipline";

export default function CalibrationPage() {
  const { data: disciplineStatus, isLoading: disciplineLoading } =
    useDisciplineStatus();
  const { data: calibrationReport, isLoading: calibrationLoading } =
    useCalibrationReport();
  const { data: marketMakerStatus, isLoading: mmLoading } =
    useMarketMakerStatus();

  // Prepare calibration chart data
  const calibrationChartData = useMemo(() => {
    if (!calibrationReport?.calibration_curve) return [];
    return calibrationReport.calibration_curve.map((bucket) => ({
      bucket: `${bucket.bucket_start}-${bucket.bucket_end}%`,
      predicted: bucket.predicted_prob,
      actual: bucket.actual_win_rate * 100,
      trades: bucket.trade_count,
    }));
  }, [calibrationReport]);

  const emptyState = (msg: string) => (
    <div className="flex h-[300px] items-center justify-center rounded-lg border border-dashed">
      <div className="text-center">
        <Activity className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
        <p className="text-sm text-muted-foreground">{msg}</p>
      </div>
    </div>
  );

  const circuitBreakerCard = (
    title: string,
    isHit: boolean,
    currentValue: number,
    limit: number,
    suffix: string
  ) => (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {isHit ? (
          <AlertCircle className="h-4 w-4 text-red-500" />
        ) : (
          <CheckCircle2 className="h-4 w-4 text-green-500" />
        )}
      </CardHeader>
      <CardContent>
        <div className={`text-2xl font-bold ${isHit ? "text-red-500" : ""}`}>
          {currentValue.toFixed(2)}
          {suffix}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Limit: {limit.toFixed(2)}
          {suffix}
        </p>
      </CardContent>
    </Card>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          Discipline & Calibration
        </h1>
        <p className="text-muted-foreground">
          Real-time monitoring of risk controls and prediction calibration
        </p>
      </div>

      {/* Section 1: Discipline Engine Status */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Discipline Engine Status</h2>

        {/* Circuit Breaker Row */}
        <div className="grid gap-4 md:grid-cols-3">
          {disciplineLoading ? (
            <div className="col-span-3 flex h-[150px] items-center justify-center">
              <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : disciplineStatus ? (
            <>
              {circuitBreakerCard(
                "Daily Loss",
                disciplineStatus.circuit_breaker.daily_limit_hit,
                disciplineStatus.circuit_breaker.daily_pnl,
                disciplineStatus.circuit_breaker.daily_loss_limit,
                " $"
              )}
              {circuitBreakerCard(
                "Weekly Loss",
                disciplineStatus.circuit_breaker.weekly_limit_hit,
                disciplineStatus.circuit_breaker.weekly_pnl,
                disciplineStatus.circuit_breaker.weekly_loss_limit,
                " $"
              )}
              {circuitBreakerCard(
                "Drawdown",
                disciplineStatus.circuit_breaker.drawdown_limit_hit,
                disciplineStatus.circuit_breaker.current_drawdown_pct,
                disciplineStatus.circuit_breaker.drawdown_limit_pct,
                "%"
              )}
            </>
          ) : null}
        </div>

        {/* Rate Limits & Concentration Row */}
        <div className="grid gap-4 md:grid-cols-3">
          {disciplineStatus ? (
            <>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Orders Today
                  </CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {disciplineStatus.rate_limits.orders_today} /{" "}
                    {disciplineStatus.rate_limits.max_orders_daily}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {disciplineStatus.rate_limits.orders_this_hour} /
                    {disciplineStatus.rate_limits.max_orders_hourly} this hour
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Deployment
                  </CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {disciplineStatus.concentration.total_deployed_pct.toFixed(
                      1
                    )}
                    %
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Largest market:{" "}
                    {disciplineStatus.concentration.largest_market_pct.toFixed(
                      1
                    )}
                    %
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Cash Reserve
                  </CardTitle>
                  <Badge
                    variant={
                      disciplineStatus.cash_reserve_pct > 20
                        ? "default"
                        : "destructive"
                    }
                  >
                    {disciplineStatus.cash_reserve_pct.toFixed(1)}%
                  </Badge>
                </CardHeader>
                <CardContent>
                  <div
                    className={`text-2xl font-bold ${
                      disciplineStatus.cash_reserve_pct < 20
                        ? "text-red-500"
                        : ""
                    }`}
                  >
                    {disciplineStatus.cash_reserve_pct.toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {disciplineStatus.cash_reserve_pct > 20
                      ? "Safe"
                      : "Low cash"}
                  </p>
                </CardContent>
              </Card>
            </>
          ) : null}
        </div>
      </div>

      {/* Section 2: Calibration Chart & Strategy Stats */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Calibration Analysis</h2>

        <Card>
          <CardHeader>
            <CardTitle>Calibration Curve</CardTitle>
          </CardHeader>
          <CardContent>
            {calibrationLoading ? (
              <div className="flex h-[350px] items-center justify-center">
                <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : calibrationChartData.length === 0 ? (
              emptyState("No calibration data available")
            ) : (
              <ResponsiveContainer width="100%" height={350}>
                <ComposedChart data={calibrationChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="bucket" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `${v}%`} />
                  <Tooltip
                    formatter={(value) => [`${Number(value).toFixed(1)}%`]}
                    labelFormatter={(label) => `Probability Bucket: ${label}`}
                  />
                  {/* Perfect calibration diagonal line */}
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="hsl(0, 0%, 60%)"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Perfect Calibration"
                  />
                  {/* Actual calibration area */}
                  <defs>
                    <linearGradient id="calibGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(59, 89%, 38%)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(59, 89%, 38%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area
                    type="monotone"
                    dataKey="actual"
                    stroke="hsl(59, 89%, 38%)"
                    fill="url(#calibGradient)"
                    strokeWidth={2}
                    name="Actual Win Rate"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Strategy Stats Table */}
        <Card>
          <CardHeader>
            <CardTitle>Per-Strategy Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            {calibrationLoading ? (
              <div className="flex h-[200px] items-center justify-center">
                <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : !calibrationReport?.strategy_stats ||
              calibrationReport.strategy_stats.length === 0 ? (
              emptyState("No strategy data available")
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Strategy</TableHead>
                    <TableHead className="text-right">Trades</TableHead>
                    <TableHead className="text-right">Win Rate</TableHead>
                    <TableHead className="text-right">Avg Edge</TableHead>
                    <TableHead className="text-right">Total P&L</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {calibrationReport.strategy_stats.map((stat) => (
                    <TableRow key={stat.strategy}>
                      <TableCell className="font-medium">
                        {stat.strategy}
                      </TableCell>
                      <TableCell className="text-right">
                        {stat.total_trades}
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="outline">
                          {(stat.win_rate * 100).toFixed(1)}%
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {(stat.avg_edge * 100).toFixed(2)}%
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge
                          variant={
                            stat.total_pnl >= 0 ? "default" : "destructive"
                          }
                          className="font-mono"
                        >
                          {stat.total_pnl >= 0 ? "+" : ""}${stat.total_pnl.toFixed(2)}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Section 3: Market Maker Status */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Market Maker Status</h2>

        {/* Summary Cards */}
        <div className="grid gap-4 md:grid-cols-4">
          {mmLoading ? (
            <div className="col-span-4 flex h-[150px] items-center justify-center">
              <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : marketMakerStatus ? (
            <>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Active Markets
                  </CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {marketMakerStatus.active_markets}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Total Inventory
                  </CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {marketMakerStatus.total_inventory}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Net Delta
                  </CardTitle>
                  <TrendingDown className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {marketMakerStatus.net_delta.toFixed(2)}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Resting Orders
                  </CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {marketMakerStatus.resting_orders}
                  </div>
                </CardContent>
              </Card>
            </>
          ) : null}
        </div>

        {/* Inventory by Market Table */}
        <Card>
          <CardHeader>
            <CardTitle>Inventory by Market</CardTitle>
          </CardHeader>
          <CardContent>
            {mmLoading ? (
              <div className="flex h-[200px] items-center justify-center">
                <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : !marketMakerStatus?.inventory_by_market ||
              marketMakerStatus.inventory_by_market.length === 0 ? (
              emptyState("No inventory data available")
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Ticker</TableHead>
                    <TableHead>Title</TableHead>
                    <TableHead className="text-right">Side</TableHead>
                    <TableHead className="text-right">Contracts</TableHead>
                    <TableHead className="text-right">Position Value</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {marketMakerStatus.inventory_by_market.map((inv) => (
                    <TableRow key={`${inv.ticker}-${inv.side}`}>
                      <TableCell className="font-mono font-medium">
                        {inv.ticker}
                      </TableCell>
                      <TableCell className="text-sm">
                        {inv.title}
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="outline">{inv.side}</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {inv.contracts}
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge
                          variant={
                            inv.position_value > 0 ? "default" : "destructive"
                          }
                          className="font-mono"
                        >
                          ${inv.position_value.toFixed(2)}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
