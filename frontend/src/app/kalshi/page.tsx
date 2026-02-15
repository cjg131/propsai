"use client";

import { useState } from "react";
import { useKalshiMarkets, KalshiMarket } from "@/lib/hooks/use-kalshi";
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
  BarChart3,
  Search,
  TrendingUp,
  TrendingDown,
  Activity,
  RefreshCw,
  ExternalLink,
} from "lucide-react";

const PROP_TYPE_COLORS: Record<string, string> = {
  points: "bg-blue-500/10 text-blue-600 border-blue-500/30",
  rebounds: "bg-green-500/10 text-green-600 border-green-500/30",
  assists: "bg-purple-500/10 text-purple-600 border-purple-500/30",
  threes: "bg-orange-500/10 text-orange-600 border-orange-500/30",
};

function MarketCard({ market }: { market: KalshiMarket }) {
  const probOver = Math.round(market.implied_prob_over * 100);
  const probUnder = Math.round(market.implied_prob_under * 100);
  const spread = market.yes_ask - market.yes_bid;

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-semibold text-sm truncate">
                {market.player_name || "Unknown Player"}
              </h3>
              <Badge
                variant="outline"
                className={`text-[10px] ${PROP_TYPE_COLORS[market.prop_type] || "bg-gray-500/10 text-gray-600"}`}
              >
                {market.prop_type}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground truncate mb-2">
              {market.title}
            </p>
            {market.line && (
              <div className="flex items-center gap-1 mb-2">
                <span className="text-xs text-muted-foreground">Line:</span>
                <span className="text-sm font-bold">{market.line}</span>
              </div>
            )}
          </div>

          <a
            href={`https://kalshi.com/markets/${market.ticker.toLowerCase()}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground"
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>

        {/* Price Grid */}
        <div className="grid grid-cols-2 gap-2 mt-3">
          <div className="rounded-lg bg-emerald-500/5 border border-emerald-500/20 p-2 text-center">
            <div className="flex items-center justify-center gap-1 mb-0.5">
              <TrendingUp className="h-3 w-3 text-emerald-500" />
              <span className="text-[10px] font-medium text-emerald-600">YES / OVER</span>
            </div>
            <div className="text-lg font-bold text-emerald-600">{market.yes_ask}¢</div>
            <div className="text-[10px] text-muted-foreground">{probOver}% implied</div>
          </div>
          <div className="rounded-lg bg-red-500/5 border border-red-500/20 p-2 text-center">
            <div className="flex items-center justify-center gap-1 mb-0.5">
              <TrendingDown className="h-3 w-3 text-red-500" />
              <span className="text-[10px] font-medium text-red-600">NO / UNDER</span>
            </div>
            <div className="text-lg font-bold text-red-600">{market.no_ask}¢</div>
            <div className="text-[10px] text-muted-foreground">{probUnder}% implied</div>
          </div>
        </div>

        {/* Market Stats */}
        <div className="flex items-center justify-between mt-3 pt-2 border-t text-[10px] text-muted-foreground">
          <span>Vol: {market.volume.toLocaleString()}</span>
          <span>Spread: {spread}¢</span>
          <span>OI: {market.open_interest.toLocaleString()}</span>
          <Badge variant="outline" className="text-[9px]">
            {market.ticker}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

export default function KalshiMarketsPage() {
  const { data, isLoading, error, refetch, isFetching } = useKalshiMarkets();
  const [search, setSearch] = useState("");
  const [propFilter, setPropFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<string>("volume");

  const markets = data?.markets || [];

  // Filter
  const filtered = markets.filter((m) => {
    if (propFilter !== "all" && m.prop_type !== propFilter) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        m.player_name.toLowerCase().includes(q) ||
        m.title.toLowerCase().includes(q) ||
        m.ticker.toLowerCase().includes(q)
      );
    }
    return true;
  });

  // Sort
  const sorted = [...filtered].sort((a, b) => {
    switch (sortBy) {
      case "volume":
        return b.volume - a.volume;
      case "price_low":
        return a.yes_ask - b.yes_ask;
      case "price_high":
        return b.yes_ask - a.yes_ask;
      case "spread":
        return (a.yes_ask - a.yes_bid) - (b.yes_ask - b.yes_bid);
      default:
        return 0;
    }
  });

  // Stats
  const propTypes = [...new Set(markets.map((m) => m.prop_type))];
  const totalVolume = markets.reduce((sum, m) => sum + m.volume, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Today&apos;s Markets</h1>
          <p className="text-sm text-muted-foreground">
            NBA player prop markets on Kalshi
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-emerald-500" />
              <span className="text-sm text-muted-foreground">Markets</span>
            </div>
            <p className="text-2xl font-bold mt-1">{markets.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-blue-500" />
              <span className="text-sm text-muted-foreground">Total Volume</span>
            </div>
            <p className="text-2xl font-bold mt-1">{totalVolume.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-purple-500" />
              <span className="text-sm text-muted-foreground">Prop Types</span>
            </div>
            <p className="text-2xl font-bold mt-1">{propTypes.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Search className="h-4 w-4 text-orange-500" />
              <span className="text-sm text-muted-foreground">Showing</span>
            </div>
            <p className="text-2xl font-bold mt-1">{sorted.length}</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search player, market..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
        <Select value={propFilter} onValueChange={setPropFilter}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Prop Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Props</SelectItem>
            <SelectItem value="points">Points</SelectItem>
            <SelectItem value="rebounds">Rebounds</SelectItem>
            <SelectItem value="assists">Assists</SelectItem>
            <SelectItem value="threes">Threes</SelectItem>
          </SelectContent>
        </Select>
        <Select value={sortBy} onValueChange={setSortBy}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Sort By" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="volume">Most Volume</SelectItem>
            <SelectItem value="price_low">Cheapest YES</SelectItem>
            <SelectItem value="price_high">Most Expensive YES</SelectItem>
            <SelectItem value="spread">Tightest Spread</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Markets Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-4 h-48" />
            </Card>
          ))}
        </div>
      ) : error ? (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-destructive font-medium">Failed to load Kalshi markets</p>
            <p className="text-sm text-muted-foreground mt-1">
              {error instanceof Error ? error.message : "Unknown error"}
            </p>
            <Button variant="outline" size="sm" className="mt-4" onClick={() => refetch()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      ) : sorted.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center">
            <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
            <p className="font-medium">No markets found</p>
            <p className="text-sm text-muted-foreground mt-1">
              {markets.length === 0
                ? "No NBA player prop markets are currently open on Kalshi."
                : "Try adjusting your filters."}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sorted.map((market) => (
            <MarketCard key={market.ticker} market={market} />
          ))}
        </div>
      )}
    </div>
  );
}
