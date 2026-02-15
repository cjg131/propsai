"use client";

import { useKalshiEdges, KalshiEdge } from "@/lib/hooks/use-kalshi";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Crosshair,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ExternalLink,
  Zap,
  Target,
  BarChart3,
} from "lucide-react";

const PROP_TYPE_COLORS: Record<string, string> = {
  points: "bg-blue-500/10 text-blue-600 border-blue-500/30",
  rebounds: "bg-green-500/10 text-green-600 border-green-500/30",
  assists: "bg-purple-500/10 text-purple-600 border-purple-500/30",
  threes: "bg-orange-500/10 text-orange-600 border-orange-500/30",
};

function EdgeCard({ edge }: { edge: KalshiEdge }) {
  const isYes = edge.edge_type === "cheap_yes";
  const probOver = Math.round(edge.implied_prob_over * 100);
  const probUnder = Math.round(edge.implied_prob_under * 100);

  return (
    <Card className="hover:shadow-md transition-shadow border-l-4 border-l-emerald-500">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-semibold text-sm truncate">
                {edge.player_name || "Unknown Player"}
              </h3>
              <Badge
                variant="outline"
                className={`text-[10px] ${PROP_TYPE_COLORS[edge.prop_type] || "bg-gray-500/10 text-gray-600"}`}
              >
                {edge.prop_type}
              </Badge>
              <Badge className="text-[9px] font-bold px-1.5 py-0.5 text-white border-0 bg-emerald-500 hover:bg-emerald-600">
                EDGE
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground truncate mb-2">
              {edge.title}
            </p>
          </div>
          <a
            href={`https://kalshi.com/markets/${edge.ticker.toLowerCase()}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground"
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>

        {/* Edge Signal */}
        <div className={`rounded-lg p-3 mt-2 ${isYes ? "bg-emerald-500/5 border border-emerald-500/20" : "bg-red-500/5 border border-red-500/20"}`}>
          <div className="flex items-center gap-2 mb-1">
            {isYes ? (
              <TrendingUp className="h-4 w-4 text-emerald-500" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-500" />
            )}
            <span className={`text-sm font-bold ${isYes ? "text-emerald-600" : "text-red-600"}`}>
              {edge.recommendation}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 mt-2">
            <div>
              <span className="text-[10px] text-muted-foreground">YES Price</span>
              <p className="text-sm font-semibold">{edge.yes_ask}¢ ({probOver}%)</p>
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground">NO Price</span>
              <p className="text-sm font-semibold">{edge.no_ask}¢ ({probUnder}%)</p>
            </div>
          </div>
          {edge.line && (
            <div className="mt-2">
              <span className="text-[10px] text-muted-foreground">Line</span>
              <p className="text-sm font-semibold">{edge.line}</p>
            </div>
          )}
        </div>

        {/* Market Stats */}
        <div className="flex items-center justify-between mt-3 pt-2 border-t text-[10px] text-muted-foreground">
          <span>Vol: {edge.volume.toLocaleString()}</span>
          <span>OI: {edge.open_interest.toLocaleString()}</span>
          <Badge variant="outline" className="text-[9px]">
            {edge.ticker}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

export default function KalshiEdgesPage() {
  const { data, isLoading, error, refetch, isFetching } = useKalshiEdges();

  const edges = data?.edges || [];
  const totalScanned = data?.total_markets_scanned || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Edge Finder</h1>
          <p className="text-sm text-muted-foreground">
            Kalshi markets where our model sees mispriced contracts
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
          Scan Markets
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-emerald-500" />
              <span className="text-sm text-muted-foreground">Edges Found</span>
            </div>
            <p className="text-2xl font-bold mt-1">{edges.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-blue-500" />
              <span className="text-sm text-muted-foreground">Markets Scanned</span>
            </div>
            <p className="text-2xl font-bold mt-1">{totalScanned}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-purple-500" />
              <span className="text-sm text-muted-foreground">Hit Rate</span>
            </div>
            <p className="text-2xl font-bold mt-1">
              {totalScanned > 0 ? `${Math.round((edges.length / totalScanned) * 100)}%` : "—"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Edge List */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-4 h-52" />
            </Card>
          ))}
        </div>
      ) : error ? (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-destructive font-medium">Failed to scan for edges</p>
            <p className="text-sm text-muted-foreground mt-1">
              {error instanceof Error ? error.message : "Unknown error"}
            </p>
            <Button variant="outline" size="sm" className="mt-4" onClick={() => refetch()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      ) : edges.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center">
            <Crosshair className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
            <p className="font-medium">No edges detected</p>
            <p className="text-sm text-muted-foreground mt-1">
              {totalScanned === 0
                ? "No NBA player prop markets are currently open on Kalshi."
                : `Scanned ${totalScanned} markets — no mispriced contracts found right now.`}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {edges.map((edge) => (
            <EdgeCard key={edge.ticker} edge={edge} />
          ))}
        </div>
      )}

      {/* Methodology Note */}
      <Card className="bg-muted/50">
        <CardContent className="p-4">
          <h3 className="text-sm font-semibold mb-1">How Edge Detection Works</h3>
          <p className="text-xs text-muted-foreground">
            We compare our model&apos;s probability estimates (based on L10 game averages, line deviation,
            and historical hit rates) against Kalshi&apos;s implied probabilities (derived from YES/NO contract prices).
            When our model significantly disagrees with the market price, we flag it as a potential edge.
            Cheap contracts (≤30¢) with high model confidence are the strongest signals.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
