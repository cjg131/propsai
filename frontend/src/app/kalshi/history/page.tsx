"use client";

import { useState } from "react";
import {
  useKalshiHistory,
  useKalshiBacktest,
  KalshiSettledMarket,
} from "@/lib/hooks/use-kalshi";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  History,
  Search,
  Calendar,
  Users,
  TrendingUp,
  TrendingDown,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Target,
  DollarSign,
  Percent,
} from "lucide-react";

const PROP_COLORS: Record<string, string> = {
  points: "bg-blue-500/10 text-blue-600 border-blue-500/30",
  rebounds: "bg-green-500/10 text-green-600 border-green-500/30",
  assists: "bg-purple-500/10 text-purple-600 border-purple-500/30",
  blocks: "bg-amber-500/10 text-amber-600 border-amber-500/30",
  steals: "bg-rose-500/10 text-rose-600 border-rose-500/30",
  threes: "bg-orange-500/10 text-orange-600 border-orange-500/30",
};

const PAGE_SIZE = 100;

function ResultBadge({ result }: { result: string }) {
  if (result === "yes") {
    return (
      <Badge className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30 text-[10px]">
        YES
      </Badge>
    );
  }
  if (result === "no") {
    return (
      <Badge className="bg-red-500/10 text-red-600 border-red-500/30 text-[10px]">
        NO
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="text-[10px]">
      {result}
    </Badge>
  );
}

function BacktestSection() {
  const { data: bt, isLoading } = useKalshiBacktest();

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i} className="animate-pulse">
            <CardContent className="p-4 h-24" />
          </Card>
        ))}
      </div>
    );
  }

  if (!bt) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground text-sm">
          No backtest results available. Run backtest_kalshi.py to generate.
        </CardContent>
      </Card>
    );
  }

  const s = bt.summary;
  const pnlColor = s.total_pnl >= 0 ? "text-emerald-600" : "text-red-600";
  const roiColor = s.roi >= 0 ? "text-emerald-600" : "text-red-600";

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Backtest Results</h2>
      <p className="text-xs text-muted-foreground">
        Min edge: {bt.settings.min_edge}% &middot; Min volume: {bt.settings.min_volume} &middot; Bet size: ${bt.settings.bet_size}
      </p>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Card>
          <CardContent className="p-3">
            <div className="flex items-center gap-1.5 mb-1">
              <Target className="h-3.5 w-3.5 text-blue-500" />
              <span className="text-xs text-muted-foreground">Bets</span>
            </div>
            <p className="text-xl font-bold">{s.bets_placed}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <div className="flex items-center gap-1.5 mb-1">
              <TrendingUp className="h-3.5 w-3.5 text-emerald-500" />
              <span className="text-xs text-muted-foreground">Win Rate</span>
            </div>
            <p className="text-xl font-bold">{s.win_rate}%</p>
            <p className="text-[10px] text-muted-foreground">{s.bets_won}W - {s.bets_placed - s.bets_won}L</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <div className="flex items-center gap-1.5 mb-1">
              <DollarSign className="h-3.5 w-3.5 text-amber-500" />
              <span className="text-xs text-muted-foreground">Wagered</span>
            </div>
            <p className="text-xl font-bold">${s.total_wagered.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <div className="flex items-center gap-1.5 mb-1">
              <DollarSign className={`h-3.5 w-3.5 ${pnlColor}`} />
              <span className="text-xs text-muted-foreground">P&L</span>
            </div>
            <p className={`text-xl font-bold ${pnlColor}`}>
              {s.total_pnl >= 0 ? "+" : ""}${s.total_pnl.toLocaleString()}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <div className="flex items-center gap-1.5 mb-1">
              <Percent className={`h-3.5 w-3.5 ${roiColor}`} />
              <span className="text-xs text-muted-foreground">ROI</span>
            </div>
            <p className={`text-xl font-bold ${roiColor}`}>
              {s.roi >= 0 ? "+" : ""}{s.roi}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* By Prop Type */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">By Prop Type</CardTitle>
          </CardHeader>
          <CardContent className="p-3 pt-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Prop</TableHead>
                  <TableHead className="text-xs text-right">Record</TableHead>
                  <TableHead className="text-xs text-right">P&L</TableHead>
                  <TableHead className="text-xs text-right">ROI</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(bt.by_prop).map(([prop, d]) => (
                  <TableRow key={prop}>
                    <TableCell>
                      <Badge variant="outline" className={`text-[10px] ${PROP_COLORS[prop] || ""}`}>
                        {prop}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-right">
                      {d.wins}W-{d.bets - d.wins}L
                    </TableCell>
                    <TableCell className={`text-xs text-right font-medium ${d.pnl >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                      ${d.pnl.toLocaleString()}
                    </TableCell>
                    <TableCell className={`text-xs text-right ${d.roi >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                      {d.roi >= 0 ? "+" : ""}{d.roi}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">By Direction</CardTitle>
          </CardHeader>
          <CardContent className="p-3 pt-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Side</TableHead>
                  <TableHead className="text-xs text-right">Record</TableHead>
                  <TableHead className="text-xs text-right">P&L</TableHead>
                  <TableHead className="text-xs text-right">ROI</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(bt.by_direction).map(([dir, d]) => (
                  <TableRow key={dir}>
                    <TableCell>
                      <Badge className={dir === "YES"
                        ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30 text-[10px]"
                        : "bg-red-500/10 text-red-600 border-red-500/30 text-[10px]"
                      }>
                        {dir}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-right">
                      {d.wins}W-{d.bets - d.wins}L
                    </TableCell>
                    <TableCell className={`text-xs text-right font-medium ${d.pnl >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                      ${d.pnl.toLocaleString()}
                    </TableCell>
                    <TableCell className={`text-xs text-right ${d.roi >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                      {d.roi >= 0 ? "+" : ""}{d.roi}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function KalshiHistoryPage() {
  const [search, setSearch] = useState("");
  const [propFilter, setPropFilter] = useState<string>("all");
  const [offset, setOffset] = useState(0);
  const [tab, setTab] = useState<"markets" | "backtest">("markets");

  const { data, isLoading, error } = useKalshiHistory({
    prop_type: propFilter !== "all" ? propFilter : undefined,
    player: search || undefined,
    limit: PAGE_SIZE,
    offset,
  });

  const markets = data?.markets || [];
  const summary = data?.summary;
  const totalFiltered = data?.filtered || 0;
  const totalPages = Math.ceil(totalFiltered / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settled Markets</h1>
        <p className="text-sm text-muted-foreground">
          Historical NBA player prop markets from Kalshi
        </p>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <Card>
            <CardContent className="p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <History className="h-3.5 w-3.5 text-blue-500" />
                <span className="text-xs text-muted-foreground">Total Markets</span>
              </div>
              <p className="text-xl font-bold">{summary.total_markets.toLocaleString()}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <Users className="h-3.5 w-3.5 text-purple-500" />
                <span className="text-xs text-muted-foreground">Players</span>
              </div>
              <p className="text-xl font-bold">{summary.unique_players}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <Calendar className="h-3.5 w-3.5 text-amber-500" />
                <span className="text-xs text-muted-foreground">Game Days</span>
              </div>
              <p className="text-xl font-bold">{summary.game_days}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <TrendingUp className="h-3.5 w-3.5 text-emerald-500" />
                <span className="text-xs text-muted-foreground">YES Results</span>
              </div>
              <p className="text-xl font-bold">{summary.yes_results.toLocaleString()}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <TrendingDown className="h-3.5 w-3.5 text-red-500" />
                <span className="text-xs text-muted-foreground">NO Results</span>
              </div>
              <p className="text-xl font-bold">{summary.no_results.toLocaleString()}</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b pb-2">
        <Button
          variant={tab === "markets" ? "default" : "ghost"}
          size="sm"
          onClick={() => setTab("markets")}
        >
          <BarChart3 className="h-4 w-4 mr-1.5" />
          Settled Markets
        </Button>
        <Button
          variant={tab === "backtest" ? "default" : "ghost"}
          size="sm"
          onClick={() => setTab("backtest")}
        >
          <Target className="h-4 w-4 mr-1.5" />
          Backtest
        </Button>
      </div>

      {tab === "backtest" ? (
        <BacktestSection />
      ) : (
        <>
          {/* Filters */}
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search player..."
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setOffset(0);
                }}
                className="pl-9"
              />
            </div>
            <Select
              value={propFilter}
              onValueChange={(v) => {
                setPropFilter(v);
                setOffset(0);
              }}
            >
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Prop Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Props</SelectItem>
                <SelectItem value="points">Points</SelectItem>
                <SelectItem value="rebounds">Rebounds</SelectItem>
                <SelectItem value="assists">Assists</SelectItem>
                <SelectItem value="blocks">Blocks</SelectItem>
                <SelectItem value="steals">Steals</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Markets Table */}
          {isLoading ? (
            <Card className="animate-pulse">
              <CardContent className="p-4 h-96" />
            </Card>
          ) : error ? (
            <Card>
              <CardContent className="p-8 text-center">
                <p className="text-destructive font-medium">Failed to load history</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {error instanceof Error ? error.message : "Unknown error"}
                </p>
              </CardContent>
            </Card>
          ) : markets.length === 0 ? (
            <Card>
              <CardContent className="p-8 text-center">
                <History className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
                <p className="font-medium">No settled markets found</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Try adjusting your filters.
                </p>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs">Date</TableHead>
                      <TableHead className="text-xs">Player</TableHead>
                      <TableHead className="text-xs">Prop</TableHead>
                      <TableHead className="text-xs text-right">Line</TableHead>
                      <TableHead className="text-xs text-right">Last Price</TableHead>
                      <TableHead className="text-xs text-right">Volume</TableHead>
                      <TableHead className="text-xs text-center">Result</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {markets.map((m: KalshiSettledMarket) => (
                      <TableRow key={m.ticker}>
                        <TableCell className="text-xs text-muted-foreground">
                          {m.close_time?.slice(0, 10)}
                        </TableCell>
                        <TableCell className="text-xs font-medium">
                          {m.player_name || "—"}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className={`text-[10px] ${PROP_COLORS[m.prop_type] || ""}`}
                          >
                            {m.prop_type}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-xs text-right font-mono">
                          {m.line ?? "—"}
                        </TableCell>
                        <TableCell className="text-xs text-right font-mono">
                          {m.last_price}¢
                        </TableCell>
                        <TableCell className="text-xs text-right">
                          {m.volume.toLocaleString()}
                        </TableCell>
                        <TableCell className="text-center">
                          <ResultBadge result={m.result} />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>

              {/* Pagination */}
              <div className="flex items-center justify-between px-4 py-3 border-t">
                <p className="text-xs text-muted-foreground">
                  Showing {offset + 1}–{Math.min(offset + PAGE_SIZE, totalFiltered)} of{" "}
                  {totalFiltered.toLocaleString()}
                </p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-7 w-7"
                    disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <span className="text-xs text-muted-foreground">
                    Page {currentPage} of {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-7 w-7"
                    disabled={offset + PAGE_SIZE >= totalFiltered}
                    onClick={() => setOffset(offset + PAGE_SIZE)}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
