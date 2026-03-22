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
  const { data: discipline, isLoading: disciplineLoading, error: disciplineError } =
    useDisciplineStatus();
  const { data: calibration, isLoading: calibrationLoading, error: calibrationError } =
    useCalibrationReport();
  const { data: marketMaker, isLoading: mmLoading, error: mmError } =
    useMarketMakerStatus();

  // Prepare calibration chart data from backend's buckets dict
  const calibrationChartData = useMemo(() => {
    if (!calibration?.buckets) return [];
    return Object.entries(calibration.buckets)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([bucketName, bucket]) => ({
        bucket: bucketName,
        predicted: bucket.avg_predicted * 100,
        actual: bucket.actual_rate * 100,
        trades: bucket.count,
      }));
  }, [calibration]);

  // Prepare market maker inventory as array
  const inventoryRows = useMemo(() => {
    if (!marketMaker?.inventory_by_market) return [];
    return Object.entries(marketMaker.inventory_by_market).map(
      ([ticker, inv]) => ({
        ticker,
        yes: inv.yes,
        no: inv.no,
        total: inv.yes + inv.no,
      })
    );
  }, [marketMaker]);

  const emptyState = (msg: string) => (
    <div className="flex h-[300px] items-center justify-center rounded-lg border border-dashed">
      <div className="text-center">
        <Activity className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
        <p className="text-sm text-muted-foreground">{msg}</p>
      </div>
    </div>
  );

  const maxDrawdown = 0.15; // 15% drawdown limit
  const maxDailyLoss = -50; // $50 daily loss limit
  const maxWeeklyLoss = -100; // $100 weekly loss limit

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

        {/* Risk Metrics Row */}
        <div className="grid gap-4 md:grid-cols-3">
          {disciplineLoading ? (
            <div className="col-span-3 flex h-[150px] items-center justify-center">
              <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : disciplineError ? (
            <div className="col-span-3">
              {emptyState("Could not load discipline status")}
            </div>
          ) : discipline ? (
            <>
              {/* Daily P&L */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Daily P&L</CardTitle>
                  {discipline.daily_pnl < maxDailyLoss ? (
                    <AlertCircle className="h-4 w-4 text-red-500" />
                  ) : (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  )}
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl font-bold ${discipline.daily_pnl < 0 ? "text-red-500" : "text-green-500"}`}>
                    {discipline.daily_pnl >= 0 ? "+" : ""}${discipline.daily_pnl.toFixed(2)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Limit: ${Math.abs(maxDailyLoss).toFixed(2)} loss
                  </p>
                </CardContent>
              </Card>

              {/* Weekly P&L */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Weekly P&L</CardTitle>
                  {discipline.weekly_pnl < maxWeeklyLoss ? (
                    <AlertCircle className="h-4 w-4 text-red-500" />
                  ) : (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  )}
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl font-bold ${discipline.weekly_pnl < 0 ? "text-red-500" : "text-green-500"}`}>
                    {discipline.weekly_pnl >= 0 ? "+" : ""}${discipline.weekly_pnl.toFixed(2)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Limit: ${Math.abs(maxWeeklyLoss).toFixed(2)} loss
                  </p>
                </CardContent>
              </Card>

              {/* Drawdown */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Drawdown</CardTitle>
                  {discipline.drawdown_pct > maxDrawdown ? (
                    <AlertCircle className="h-4 w-4 text-red-500" />
                  ) : (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  )}
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl font-bold ${discipline.drawdown_pct > maxDrawdown ? "text-red-500" : ""}`}>
                    {(discipline.drawdown_pct * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Limit: {(maxDrawdown * 100).toFixed(0)}%
                  </p>
                </CardContent>
              </Card>
            </>
          ) : null}
        </div>

        {/* Orders & Deployment Row */}
        <div className="grid gap-4 md:grid-cols-3">
          {discipline ? (
            <>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Orders</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {discipline.orders_last_day}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {discipline.orders_last_hour} this hour &middot; {discipline.positions_count} positions
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Deployment</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {(discipline.deployment_pct * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    ${discipline.total_deployed.toFixed(2)} of ${discipline.bankroll.toFixed(2)}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Cash Available</CardTitle>
                  <Badge
                    variant={
                      discipline.cash_available > discipline.bankroll * 0.2
                        ? "default"
                        : "destructive"
                    }
                  >
                    {discipline.is_halted ? "HALTED" : "Active"}
                  </Badge>
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl font-bold ${discipline.cash_available < discipline.bankroll * 0.2 ? "text-red-500" : ""}`}>
                    ${discipline.cash_available.toFixed(2)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {discipline.halt_reason || "No halt"}
                  </p>
                </CardContent>
              </Card>
            </>
          ) : null}
        </div>
      </div>

      {/* Section 2: Calibration Chart */}
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
            ) : calibrationError || calibration?.error ? (
              emptyState(calibration?.error || "Not enough trades yet for calibration (need 20+)")
            ) : calibrationChartData.length === 0 ? (
              emptyState("No calibration data available yet")
            ) : (
              <ResponsiveContainer width="100%" height={350}>
                <ComposedChart data={calibrationChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="bucket" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `${v}%`} />
                  <Tooltip
                    formatter={(value: unknown) => [`${Number(value).toFixed(1)}%`]}
                    labelFormatter={(label) => `Bucket: ${label}`}
                  />
                  {/* Perfect calibration diagonal */}
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="hsl(0, 0%, 60%)"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Perfect Calibration"
                  />
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

        {/* Calibration Summary */}
        {calibration && !calibration.error && (
          <div className="grid gap-4 md:grid-cols-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Total Predictions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{calibration.total_predictions}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Overall Win Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(calibration.overall_win_rate * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Avg Edge Claimed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(calibration.avg_edge_claimed * 100).toFixed(2)}%
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${calibration.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {calibration.total_pnl >= 0 ? "+" : ""}${calibration.total_pnl.toFixed(2)}
                </div>
              </CardContent>
            </Card>
          </div>
        )}
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
          ) : mmError ? (
            <div className="col-span-4">
              {emptyState("Could not load market maker status")}
            </div>
          ) : marketMaker ? (
            <>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Markets</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{marketMaker.active_markets}</div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Inventory</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{marketMaker.total_inventory}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Yes: {marketMaker.total_yes} &middot; No: {marketMaker.total_no}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Net Delta</CardTitle>
                  <TrendingDown className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{marketMaker.net_delta}</div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Resting Orders</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{marketMaker.resting_orders}</div>
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
            ) : inventoryRows.length === 0 ? (
              emptyState("No inventory data available")
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Ticker</TableHead>
                    <TableHead className="text-right">Yes Contracts</TableHead>
                    <TableHead className="text-right">No Contracts</TableHead>
                    <TableHead className="text-right">Total</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {inventoryRows.map((inv) => (
                    <TableRow key={inv.ticker}>
                      <TableCell className="font-mono font-medium">
                        {inv.ticker}
                      </TableCell>
                      <TableCell className="text-right">{inv.yes}</TableCell>
                      <TableCell className="text-right">{inv.no}</TableCell>
                      <TableCell className="text-right">
                        <Badge variant="outline" className="font-mono">
                          {inv.total}
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
