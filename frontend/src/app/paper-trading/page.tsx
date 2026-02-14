"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  Target,
  CheckCircle2,
  XCircle,
  Clock,
  RotateCcw,
  Zap,
  RefreshCw,
  Flame,
  CalendarDays,
} from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

interface PaperTrade {
  id: number;
  player_name: string;
  team: string;
  opponent: string;
  game_date: string;
  prop_type: string;
  line: number;
  predicted_value: number;
  recommended_bet: string;
  confidence_score: number;
  stake: number;
  actual_value: number | null;
  result: string;
  profit: number;
  created_at: string;
}

interface PaperTradingStatus {
  status: string;
  has_table: boolean;
  total_trades: number;
  pending: number;
  wins: number;
  losses: number;
  pushes: number;
  hit_rate: number;
  total_profit: number;
  total_wagered: number;
  current_bankroll: number;
  starting_bankroll: number;
  roi: number;
  today_trades: number;
  today_pending: number;
  streak: number;
  streak_type: string;
  recent_trades: PaperTrade[];
  message?: string;
}

const PROP_LABELS: Record<string, string> = {
  points: "PTS", rebounds: "REB", assists: "AST", threes: "3PM",
  steals: "STL", blocks: "BLK", turnovers: "TO",
};

export default function PaperTradingPage() {
  const queryClient = useQueryClient();
  const [betDate, setBetDate] = useState(
    new Date().toISOString().slice(0, 10)
  );

  const { data: status, isLoading } = useQuery<PaperTradingStatus>({
    queryKey: ["paper-trading", "status"],
    queryFn: () => api.get<PaperTradingStatus>("/api/bets/paper-trading/status"),
    refetchInterval: 30000,
  });

  const placeMutation = useMutation({
    mutationFn: (targetDate: string) =>
      api.post(`/api/bets/paper-trading/place?target_date=${targetDate}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["paper-trading"] }),
  });

  const resolveMutation = useMutation({
    mutationFn: (targetDate?: string) =>
      api.post(
        targetDate
          ? `/api/bets/paper-trading/resolve?target_date=${targetDate}`
          : "/api/bets/paper-trading/resolve"
      ),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["paper-trading"] }),
  });

  const resetMutation = useMutation({
    mutationFn: () => api.post("/api/bets/paper-trading/reset"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["paper-trading"] }),
  });

  const hasTable = status?.has_table ?? false;
  const noTable = status && !hasTable;
  const trades = status?.recent_trades ?? [];
  const bankroll = status?.current_bankroll ?? 1000;
  const profit = status?.total_profit ?? 0;
  const wins = status?.wins ?? 0;
  const losses = status?.losses ?? 0;
  const pending = status?.pending ?? 0;
  const hitRate = status?.hit_rate ?? 0;
  const roi = status?.roi ?? 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Paper Trading</h1>
          <p className="text-muted-foreground">
            Virtual bankroll tracking — validate model predictions without real money
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5">
            <CalendarDays className="h-4 w-4 text-muted-foreground" />
            <Input
              type="date"
              value={betDate}
              onChange={(e) => setBetDate(e.target.value)}
              className="h-8 w-[140px] text-xs"
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => resolveMutation.mutate(betDate)}
            disabled={resolveMutation.isPending}
          >
            {resolveMutation.isPending ? (
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <CheckCircle2 className="mr-2 h-4 w-4" />
            )}
            Resolve Trades
          </Button>
          <Button
            size="sm"
            onClick={() => placeMutation.mutate(betDate)}
            disabled={placeMutation.isPending}
          >
            {placeMutation.isPending ? (
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Zap className="mr-2 h-4 w-4" />
            )}
            Place Bets
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              if (confirm("Reset all paper trades? This cannot be undone.")) {
                resetMutation.mutate();
              }
            }}
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {noTable && (
        <Card className="border-amber-500/50 bg-amber-500/5">
          <CardContent className="pt-6">
            <p className="text-sm text-amber-600">
              Paper trades table not found. Run the migration SQL in Supabase:
              <code className="ml-1 text-xs bg-muted px-1 py-0.5 rounded">
                backend/scripts/migrate_paper_trades.sql
              </code>
            </p>
          </CardContent>
        </Card>
      )}

      {placeMutation.isSuccess && (
        <Card className="border-green-500/50 bg-green-500/5">
          <CardContent className="pt-6">
            <p className="text-sm text-green-600">
              {(placeMutation.data as any)?.trades_placed ?? 0} virtual bets placed!
              Total staked: ${(placeMutation.data as any)?.total_staked?.toFixed(2) ?? "0.00"}
            </p>
          </CardContent>
        </Card>
      )}

      {resolveMutation.isSuccess && (
        <Card className="border-blue-500/50 bg-blue-500/5">
          <CardContent className="pt-6">
            <p className="text-sm text-blue-600">
              Resolved {(resolveMutation.data as any)?.resolved ?? 0} trades.
              W: {(resolveMutation.data as any)?.wins ?? 0} / L: {(resolveMutation.data as any)?.losses ?? 0} /
              P&L: ${(resolveMutation.data as any)?.profit?.toFixed(2) ?? "0.00"}
            </p>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Virtual Bankroll</CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${bankroll.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </div>
            <p className="text-xs text-muted-foreground">
              Started at $1,000.00
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Paper P&L</CardTitle>
            {profit >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-500" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-500" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${profit >= 0 ? "text-green-500" : "text-red-500"}`}>
              {profit >= 0 ? "+" : ""}${profit.toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground">
              ROI: {roi >= 0 ? "+" : ""}{roi.toFixed(1)}%
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Record</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{wins}-{losses}</div>
            <p className="text-xs text-muted-foreground">
              {(hitRate * 100).toFixed(1)}% hit rate
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{pending}</div>
            <p className="text-xs text-muted-foreground">
              {status?.today_trades ?? 0} today
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Streak</CardTitle>
            <Flame className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {status?.streak ?? 0}{status?.streak_type === "win" ? "W" : status?.streak_type === "loss" ? "L" : ""}
            </div>
            <p className="text-xs text-muted-foreground">
              {status?.total_trades ?? 0} total trades
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              Recent Paper Trades
              <Badge variant="outline" className="text-xs font-normal">
                {trades.length} shown
              </Badge>
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex h-[300px] items-center justify-center">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : trades.length === 0 ? (
            <div className="flex h-[350px] items-center justify-center rounded-lg border border-dashed">
              <div className="text-center">
                <Wallet className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
                <p className="text-lg font-medium text-muted-foreground">
                  No paper trades yet
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  Click &quot;Place Today&apos;s Bets&quot; to auto-place virtual bets on predictions
                </p>
              </div>
            </div>
          ) : (
            <div className="max-h-[500px] overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Status</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead>Player</TableHead>
                    <TableHead>Prop</TableHead>
                    <TableHead className="text-center">Pick</TableHead>
                    <TableHead className="text-right">Line</TableHead>
                    <TableHead className="text-right">Predicted</TableHead>
                    <TableHead className="text-right">Actual</TableHead>
                    <TableHead className="text-right">Stake</TableHead>
                    <TableHead className="text-right">Profit</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {trades.map((t) => (
                    <TableRow key={t.id}>
                      <TableCell>
                        {t.result === "win" ? (
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                        ) : t.result === "loss" ? (
                          <XCircle className="h-4 w-4 text-red-500" />
                        ) : (
                          <Clock className="h-4 w-4 text-muted-foreground" />
                        )}
                      </TableCell>
                      <TableCell className="text-xs">{t.game_date}</TableCell>
                      <TableCell className="font-medium text-sm">
                        {t.player_name}
                        {t.team && (
                          <span className="text-xs text-muted-foreground ml-1">
                            ({t.team})
                          </span>
                        )}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="text-[10px]">
                          {PROP_LABELS[t.prop_type] || t.prop_type}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge
                          variant={t.recommended_bet === "over" ? "default" : "secondary"}
                          className={`text-xs ${
                            t.recommended_bet === "over"
                              ? "bg-green-600"
                              : "bg-red-600 text-white"
                          }`}
                        >
                          {t.recommended_bet?.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">{t.line}</TableCell>
                      <TableCell className="text-right font-mono text-sm">{t.predicted_value?.toFixed(1)}</TableCell>
                      <TableCell className="text-right font-mono text-sm font-bold">
                        {t.actual_value != null ? t.actual_value : "—"}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">${t.stake?.toFixed(2)}</TableCell>
                      <TableCell className={`text-right font-mono text-sm font-bold ${
                        t.result === "win" ? "text-green-500" : t.result === "loss" ? "text-red-500" : ""
                      }`}>
                        {t.result === "pending" ? "—" : `${t.profit >= 0 ? "+" : ""}$${t.profit?.toFixed(2)}`}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
