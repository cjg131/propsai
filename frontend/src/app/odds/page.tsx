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
import { RefreshCw, ArrowRightLeft, Star } from "lucide-react";
import { api } from "@/lib/api";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

interface BookOdds {
  sportsbook: string;
  line: number;
  over_odds: number;
  under_odds: number;
  last_updated: string | null;
}

interface OddsComparison {
  player_id: string;
  player_name: string;
  team: string;
  opponent: string;
  game_id: string;
  prop_type: string;
  books: BookOdds[];
  best_over_book: string | null;
  best_over_odds: number | null;
  best_under_book: string | null;
  best_under_odds: number | null;
  opening_line: number | null;
  current_consensus_line: number | null;
}

function OddsCell({
  odds,
  isBest,
}: {
  odds: number;
  isBest: boolean;
}) {
  const formatted = odds > 0 ? `+${odds}` : `${odds}`;
  return (
    <span
      className={`font-mono text-sm ${
        isBest ? "text-green-500 font-bold" : "text-muted-foreground"
      }`}
    >
      {formatted}
    </span>
  );
}

export default function OddsPage() {
  const [propFilter, setPropFilter] = useState("all");
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["odds", "compare", propFilter],
    queryFn: () => {
      const params = propFilter !== "all" ? `?prop_type=${propFilter}` : "";
      return api.get<{ comparisons: OddsComparison[]; total: number }>(
        `/api/odds/compare${params}`
      );
    },
  });

  const refreshMutation = useMutation({
    mutationFn: () => api.post("/api/odds/refresh"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["odds"] }),
  });

  const comparisons = data?.comparisons ?? [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Odds Comparison
          </h1>
          <p className="text-muted-foreground">
            Compare prop lines across all available sportsbooks
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refreshMutation.mutate()}
          disabled={refreshMutation.isPending}
        >
          <RefreshCw
            className={`mr-2 h-4 w-4 ${refreshMutation.isPending ? "animate-spin" : ""}`}
          />
          {refreshMutation.isPending ? "Refreshing..." : "Refresh Odds"}
        </Button>
      </div>

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <CardTitle className="flex items-center gap-2">
              <ArrowRightLeft className="h-5 w-5" />
              Side-by-Side Odds
            </CardTitle>
            <Select value={propFilter} onValueChange={setPropFilter}>
              <SelectTrigger className="w-[140px] h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Props</SelectItem>
                <SelectItem value="points">Points</SelectItem>
                <SelectItem value="rebounds">Rebounds</SelectItem>
                <SelectItem value="assists">Assists</SelectItem>
                <SelectItem value="threes">3-Pointers</SelectItem>
                <SelectItem value="steals">Steals</SelectItem>
                <SelectItem value="blocks">Blocks</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex h-[400px] items-center justify-center">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : comparisons.length === 0 ? (
            <div className="flex h-[400px] items-center justify-center rounded-lg border border-dashed">
              <div className="text-center">
                <ArrowRightLeft className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
                <p className="text-lg font-medium text-muted-foreground">
                  No odds data available
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  Click &quot;Refresh Odds&quot; to load today&apos;s prop lines
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {comparisons.map((comp) => (
                <div key={`${comp.player_id}-${comp.prop_type}-${comp.game_id}`} className="rounded-lg border p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <span className="font-semibold">{comp.player_name}</span>
                      <span className="text-sm text-muted-foreground ml-2">
                        {comp.team}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="capitalize text-xs">
                        {comp.prop_type}
                      </Badge>
                      {comp.current_consensus_line !== null && (
                        <Badge variant="secondary" className="font-mono text-xs">
                          Line: {comp.current_consensus_line}
                        </Badge>
                      )}
                    </div>
                  </div>
                  <div className="rounded-md border overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[140px]">Sportsbook</TableHead>
                          <TableHead className="text-center">Line</TableHead>
                          <TableHead className="text-center">Over</TableHead>
                          <TableHead className="text-center">Under</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {comp.books.map((book) => (
                          <TableRow key={book.sportsbook}>
                            <TableCell className="font-medium text-sm">
                              {book.sportsbook}
                              {book.sportsbook === comp.best_over_book && (
                                <Star className="inline h-3 w-3 ml-1 text-yellow-400 fill-yellow-400" />
                              )}
                            </TableCell>
                            <TableCell className="text-center font-mono text-sm">
                              {book.line}
                            </TableCell>
                            <TableCell className="text-center">
                              <OddsCell
                                odds={book.over_odds}
                                isBest={book.sportsbook === comp.best_over_book}
                              />
                            </TableCell>
                            <TableCell className="text-center">
                              <OddsCell
                                odds={book.under_odds}
                                isBest={book.sportsbook === comp.best_under_book}
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
