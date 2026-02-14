"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
  Plus,
  Trophy,
  Percent,
  TrendingUp,
  DollarSign,
  Flame,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Clock,
  Minus,
} from "lucide-react";
import { useBets, useBetSummary, type Bet } from "@/lib/hooks/use-bets";

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "won":
      return <CheckCircle2 className="h-4 w-4 text-green-500" />;
    case "lost":
      return <XCircle className="h-4 w-4 text-red-500" />;
    case "push":
      return <Minus className="h-4 w-4 text-yellow-500" />;
    default:
      return <Clock className="h-4 w-4 text-muted-foreground" />;
  }
}

function BetRow({ bet }: { bet: Bet }) {
  return (
    <TableRow>
      <TableCell>
        <div className="flex items-center gap-2">
          <StatusIcon status={bet.status} />
          <div>
            <span className="font-medium text-sm">{bet.player_name}</span>
            <div className="text-xs text-muted-foreground">
              {bet.team} vs {bet.opponent || "TBD"}
            </div>
          </div>
        </div>
      </TableCell>
      <TableCell>
        <Badge variant="outline" className="text-xs capitalize">
          {bet.prop_type}
        </Badge>
      </TableCell>
      <TableCell className="text-center font-mono text-sm">
        {bet.line}
      </TableCell>
      <TableCell className="text-center">
        <Badge
          variant={bet.bet_type === "over" ? "default" : "secondary"}
          className={`text-xs ${
            bet.bet_type === "over"
              ? "bg-green-600 hover:bg-green-700"
              : "bg-red-600 hover:bg-red-700 text-white"
          }`}
        >
          {bet.bet_type.toUpperCase()}
        </Badge>
      </TableCell>
      <TableCell className="text-center font-mono text-sm">
        {bet.odds > 0 ? `+${bet.odds}` : bet.odds}
      </TableCell>
      <TableCell className="text-center font-mono text-sm">
        ${bet.stake.toFixed(0)}
      </TableCell>
      <TableCell className="text-center text-xs text-muted-foreground">
        {bet.sportsbook}
      </TableCell>
      <TableCell className="text-center">
        {bet.actual_value !== null ? (
          <span className="font-mono text-sm">{bet.actual_value}</span>
        ) : (
          <span className="text-xs text-muted-foreground">--</span>
        )}
      </TableCell>
      <TableCell className="text-center">
        {bet.profit !== null ? (
          <span
            className={`font-mono text-sm font-semibold ${
              bet.profit > 0
                ? "text-green-500"
                : bet.profit < 0
                ? "text-red-500"
                : "text-muted-foreground"
            }`}
          >
            {bet.profit > 0 ? "+" : ""}${bet.profit.toFixed(2)}
          </span>
        ) : (
          <Badge variant="outline" className="text-xs">
            Pending
          </Badge>
        )}
      </TableCell>
    </TableRow>
  );
}

export default function BetsPage() {
  const [statusFilter, setStatusFilter] = useState("all");
  const [propFilter, setPropFilter] = useState("all");

  const { data: summary } = useBetSummary();
  const { data: betsData, isLoading } = useBets({
    status: statusFilter === "all" ? undefined : statusFilter,
    prop_type: propFilter === "all" ? undefined : propFilter,
  });
  const bets = betsData?.bets ?? [];

  const record = summary
    ? `${summary.wins}-${summary.losses}${summary.pushes ? `-${summary.pushes}` : ""}`
    : "0-0";
  const winRate = summary ? `${(summary.win_rate * 100).toFixed(1)}%` : "--%";
  const roi = summary ? `${summary.roi > 0 ? "+" : ""}${summary.roi.toFixed(1)}%` : "--%";
  const profit = summary
    ? `${summary.total_profit >= 0 ? "+" : ""}$${summary.total_profit.toFixed(2)}`
    : "$0.00";
  const streakText = summary && summary.current_streak > 0
    ? `${summary.current_streak}${summary.streak_type === "won" ? "W" : "L"}`
    : "--";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Bet Tracker</h1>
          <p className="text-muted-foreground">
            Log and track all your bets, W/L record, ROI, and streaks
          </p>
        </div>
        <Button size="sm">
          <Plus className="mr-2 h-4 w-4" />
          Log Bet
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Record</CardTitle>
            <Trophy className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{record}</div>
            <p className="text-xs text-muted-foreground">
              {summary?.pending ?? 0} pending
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <Percent className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{winRate}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ROI</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${summary && summary.roi > 0 ? "text-green-500" : summary && summary.roi < 0 ? "text-red-500" : ""}`}>
              {roi}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Profit/Loss</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${summary && summary.total_profit > 0 ? "text-green-500" : summary && summary.total_profit < 0 ? "text-red-500" : ""}`}>
              {profit}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Streak</CardTitle>
            <Flame className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${summary?.streak_type === "won" ? "text-green-500" : summary?.streak_type === "lost" ? "text-red-500" : ""}`}>
              {streakText}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <CardTitle>Bet History</CardTitle>
            <div className="flex gap-2">
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-[130px] h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="won">Won</SelectItem>
                  <SelectItem value="lost">Lost</SelectItem>
                  <SelectItem value="push">Push</SelectItem>
                </SelectContent>
              </Select>
              <Select value={propFilter} onValueChange={setPropFilter}>
                <SelectTrigger className="w-[130px] h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Props</SelectItem>
                  <SelectItem value="points">Points</SelectItem>
                  <SelectItem value="rebounds">Rebounds</SelectItem>
                  <SelectItem value="assists">Assists</SelectItem>
                  <SelectItem value="threes">3-Pointers</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex h-[300px] items-center justify-center">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : bets.length === 0 ? (
            <div className="flex h-[300px] items-center justify-center rounded-lg border border-dashed">
              <div className="text-center">
                <Trophy className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
                <p className="text-lg font-medium text-muted-foreground">
                  No bets logged yet
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  Click &quot;Log Bet&quot; to start tracking your performance
                </p>
              </div>
            </div>
          ) : (
            <div className="rounded-md border overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[200px]">Player</TableHead>
                    <TableHead>Prop</TableHead>
                    <TableHead className="text-center">Line</TableHead>
                    <TableHead className="text-center">Pick</TableHead>
                    <TableHead className="text-center">Odds</TableHead>
                    <TableHead className="text-center">Stake</TableHead>
                    <TableHead className="text-center">Book</TableHead>
                    <TableHead className="text-center">Actual</TableHead>
                    <TableHead className="text-center">P&L</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {bets.map((bet) => (
                    <BetRow key={bet.id} bet={bet} />
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
