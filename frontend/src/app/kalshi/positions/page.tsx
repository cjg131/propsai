"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Wallet,
  DollarSign,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Activity,
} from "lucide-react";

interface Position {
  ticker: string;
  title: string;
  side: string;
  strategy: string;
  signal_source: string;
  contracts: number;
  avg_entry_cents: number;
  total_cost: number;
  total_fees: number;
  max_risk: number;
  max_profit: number;
  avg_our_prob: number;
  avg_entry_kalshi_prob: number;
  avg_entry_edge: number;
  first_entry: string;
  last_entry: string;
  num_fills: number;
  paper_mode: boolean;
  current_yes_ask: number | null;
  current_no_ask: number | null;
  current_yes_bid: number | null;
  current_no_bid: number | null;
  mark_price_cents: number | null;
  unrealized_pnl: number | null;
  current_edge: number | null;
  status: string;
}

interface PositionsResponse {
  positions: Position[];
  total: number;
  total_cost: number;
  total_unrealized_pnl: number;
  total_max_risk: number;
}

function useAgentPositions() {
  return useQuery<PositionsResponse>({
    queryKey: ["agent", "positions"],
    queryFn: () => api.get("/api/kalshi/agent/positions"),
    refetchInterval: 30_000,
    staleTime: 15_000,
    retry: 1,
  });
}

export default function KalshiPositionsPage() {
  const { data, isLoading, error } = useAgentPositions();

  const positions = data?.positions || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">My Positions</h1>
        <p className="text-sm text-muted-foreground">
          Open positions from the trading agent with live market data
        </p>
      </div>

      {/* Summary Cards */}
      {data && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Wallet className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">
                  Open Positions
                </span>
              </div>
              <p className="text-2xl font-bold mt-1">{data.total}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <DollarSign className="h-4 w-4 text-emerald-500" />
                <span className="text-sm text-muted-foreground">
                  Total Cost
                </span>
              </div>
              <p className="text-2xl font-bold mt-1">
                ${data.total_cost.toFixed(2)}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                {data.total_unrealized_pnl >= 0 ? (
                  <TrendingUp className="h-4 w-4 text-emerald-500" />
                ) : (
                  <TrendingDown className="h-4 w-4 text-red-500" />
                )}
                <span className="text-sm text-muted-foreground">
                  Unrealized P&L
                </span>
              </div>
              <p
                className={`text-2xl font-bold mt-1 ${
                  data.total_unrealized_pnl >= 0
                    ? "text-emerald-500"
                    : "text-red-500"
                }`}
              >
                {data.total_unrealized_pnl >= 0 ? "+" : ""}$
                {data.total_unrealized_pnl.toFixed(2)}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-amber-500" />
                <span className="text-sm text-muted-foreground">
                  Max Risk
                </span>
              </div>
              <p className="text-2xl font-bold mt-1">
                ${data.total_max_risk.toFixed(2)}
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Positions Table */}
      {isLoading ? (
        <Card>
          <CardContent className="flex h-[300px] items-center justify-center">
            <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
          </CardContent>
        </Card>
      ) : error ? (
        <Card>
          <CardContent className="p-8 text-center">
            <Activity className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
            <p className="text-sm text-muted-foreground">
              Could not load positions. Is the agent running?
            </p>
          </CardContent>
        </Card>
      ) : positions.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center">
            <Wallet className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
            <p className="font-medium">No open positions</p>
            <p className="text-sm text-muted-foreground mt-1">
              The agent will open positions when it finds profitable edges.
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Open Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Market</TableHead>
                  <TableHead>Strategy</TableHead>
                  <TableHead className="text-right">Side</TableHead>
                  <TableHead className="text-right">Qty</TableHead>
                  <TableHead className="text-right">Entry</TableHead>
                  <TableHead className="text-right">Mark</TableHead>
                  <TableHead className="text-right">Cost</TableHead>
                  <TableHead className="text-right">Unreal. P&L</TableHead>
                  <TableHead className="text-right">Edge</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.map((pos) => (
                  <TableRow key={`${pos.ticker}-${pos.side}`}>
                    <TableCell>
                      <div>
                        <p className="font-medium text-sm truncate max-w-[250px]">
                          {pos.title || pos.ticker}
                        </p>
                        <p className="text-xs text-muted-foreground font-mono">
                          {pos.ticker}
                        </p>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="text-[10px]">
                        {pos.strategy}
                      </Badge>
                      {pos.paper_mode && (
                        <Badge
                          variant="secondary"
                          className="text-[10px] ml-1"
                        >
                          PAPER
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge
                        variant={
                          pos.side === "yes" ? "default" : "destructive"
                        }
                        className="text-[10px]"
                      >
                        {pos.side.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {pos.contracts}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {pos.avg_entry_cents}¢
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {pos.mark_price_cents != null
                        ? `${pos.mark_price_cents}¢`
                        : "—"}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      ${pos.total_cost.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right">
                      {pos.unrealized_pnl != null ? (
                        <Badge
                          variant={
                            pos.unrealized_pnl >= 0
                              ? "default"
                              : "destructive"
                          }
                          className="font-mono"
                        >
                          {pos.unrealized_pnl >= 0 ? "+" : ""}$
                          {pos.unrealized_pnl.toFixed(2)}
                        </Badge>
                      ) : (
                        "—"
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {pos.current_edge != null ? (
                        <span
                          className={`font-mono text-sm ${
                            pos.current_edge > 0
                              ? "text-emerald-500"
                              : "text-red-500"
                          }`}
                        >
                          {(pos.current_edge * 100).toFixed(1)}%
                        </span>
                      ) : (
                        "—"
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
