"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  Search,
  User,
  TrendingUp,
  Activity,
  Brain,
  RefreshCw,
  AlertTriangle,
} from "lucide-react";
import { useSearchPlayers, useScoutingReport, type Player } from "@/lib/hooks/use-players";

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="text-center p-3 rounded-lg bg-muted/50">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-xl font-bold mt-0.5">{value}</div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

function PlayerCard({
  player,
  isSelected,
  onClick,
}: {
  player: Player;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
        isSelected ? "bg-primary/10 border border-primary" : "hover:bg-muted/50 border border-transparent"
      }`}
    >
      <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center">
        <User className="h-5 w-5 text-muted-foreground" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-semibold text-sm truncate">{player.name}</div>
        <div className="text-xs text-muted-foreground">
          {player.team} &middot; {player.position}
          {player.jersey_number && ` #${player.jersey_number}`}
        </div>
      </div>
      <div className="flex gap-1">
        {player.is_starter && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            Starter
          </Badge>
        )}
        {player.is_rookie && (
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
            Rookie
          </Badge>
        )}
      </div>
    </div>
  );
}

export default function PlayersPage() {
  const [query, setQuery] = useState("");
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);

  const { data: searchData, isLoading: searching } = useSearchPlayers(
    query.length >= 2 ? query : undefined
  );
  const players = searchData?.players ?? [];

  const { data: report, isLoading: loadingReport } = useScoutingReport(
    selectedPlayer?.id ?? ""
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Player Research</h1>
        <p className="text-muted-foreground">
          Deep dive into player stats, trends, matchups, and AI scouting reports
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[350px_1fr]">
        {/* Search Panel */}
        <div className="space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search players..."
              className="pl-9"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          <div className="space-y-1 max-h-[600px] overflow-y-auto">
            {searching && (
              <div className="flex items-center justify-center py-8">
                <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            )}
            {!searching && query.length >= 2 && players.length === 0 && (
              <div className="text-center py-8 text-sm text-muted-foreground">
                No players found for &quot;{query}&quot;
              </div>
            )}
            {!searching && query.length < 2 && (
              <div className="text-center py-8 text-sm text-muted-foreground">
                Type at least 2 characters to search
              </div>
            )}
            {players.map((player) => (
              <PlayerCard
                key={player.id}
                player={player}
                isSelected={selectedPlayer?.id === player.id}
                onClick={() => setSelectedPlayer(player)}
              />
            ))}
          </div>
        </div>

        {/* Player Detail Panel */}
        <div className="space-y-4">
          {!selectedPlayer ? (
            <Card>
              <CardContent className="flex h-[500px] items-center justify-center">
                <div className="text-center">
                  <User className="h-12 w-12 mx-auto text-muted-foreground mb-3" />
                  <p className="text-lg font-medium text-muted-foreground">
                    Select a player to view their profile
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Search and click a player to see stats, trends, and AI scouting report
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Player Header */}
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-start gap-4">
                    <div className="h-16 w-16 rounded-full bg-muted flex items-center justify-center shrink-0">
                      <User className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <div className="flex-1">
                      <h2 className="text-2xl font-bold">{selectedPlayer.name}</h2>
                      <div className="flex items-center gap-2 mt-1 text-sm text-muted-foreground">
                        <span>{selectedPlayer.team}</span>
                        <span>&middot;</span>
                        <span>{selectedPlayer.position}</span>
                        {selectedPlayer.jersey_number && (
                          <>
                            <span>&middot;</span>
                            <span>#{selectedPlayer.jersey_number}</span>
                          </>
                        )}
                      </div>
                      <div className="flex gap-2 mt-2">
                        {selectedPlayer.is_starter && <Badge>Starter</Badge>}
                        {selectedPlayer.is_rookie && <Badge variant="secondary">Rookie</Badge>}
                        {selectedPlayer.is_recently_traded && (
                          <Badge variant="destructive" className="gap-1">
                            <AlertTriangle className="h-3 w-3" />
                            Recently Traded
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Season Stats */}
              {selectedPlayer.stats && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Activity className="h-4 w-4" />
                      Season Averages
                      <Badge variant="outline" className="text-xs">
                        {selectedPlayer.stats.games_played} GP
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
                      <StatCard label="PTS" value={selectedPlayer.stats.season_avg_points.toFixed(1)} />
                      <StatCard label="REB" value={selectedPlayer.stats.season_avg_rebounds.toFixed(1)} />
                      <StatCard label="AST" value={selectedPlayer.stats.season_avg_assists.toFixed(1)} />
                      <StatCard label="3PM" value={selectedPlayer.stats.season_avg_threes.toFixed(1)} />
                      <StatCard label="STL" value={selectedPlayer.stats.season_avg_steals.toFixed(1)} />
                      <StatCard label="BLK" value={selectedPlayer.stats.season_avg_blocks.toFixed(1)} />
                      <StatCard label="TO" value={selectedPlayer.stats.season_avg_turnovers.toFixed(1)} />
                      <StatCard label="MIN" value={selectedPlayer.stats.season_avg_minutes.toFixed(1)} />
                    </div>

                    <Separator className="my-4" />

                    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                      <div>
                        <h4 className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-1">
                          <TrendingUp className="h-3 w-3" />
                          Last 5 Games
                        </h4>
                        <div className="grid grid-cols-3 gap-2">
                          <StatCard label="PTS" value={selectedPlayer.stats.last5_avg_points.toFixed(1)} />
                          <StatCard label="REB" value={selectedPlayer.stats.last5_avg_rebounds.toFixed(1)} />
                          <StatCard label="AST" value={selectedPlayer.stats.last5_avg_assists.toFixed(1)} />
                        </div>
                      </div>
                      <div>
                        <h4 className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-1">
                          <TrendingUp className="h-3 w-3" />
                          Last 10 Games
                        </h4>
                        <div className="grid grid-cols-3 gap-2">
                          <StatCard label="PTS" value={selectedPlayer.stats.last10_avg_points.toFixed(1)} />
                          <StatCard label="REB" value={selectedPlayer.stats.last10_avg_rebounds.toFixed(1)} />
                          <StatCard label="AST" value={selectedPlayer.stats.last10_avg_assists.toFixed(1)} />
                        </div>
                      </div>
                      <div>
                        <h4 className="text-xs font-semibold text-muted-foreground mb-2">
                          Usage Rate
                        </h4>
                        <StatCard label="USG%" value={`${(selectedPlayer.stats.usage_rate * 100).toFixed(1)}%`} />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* AI Scouting Report */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Brain className="h-4 w-4" />
                    AI Scouting Report
                    {report?.model_used && (
                      <Badge variant="outline" className="text-xs">
                        {report.model_used}
                      </Badge>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {loadingReport ? (
                    <div className="flex items-center justify-center py-12">
                      <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
                      <span className="ml-2 text-sm text-muted-foreground">
                        Generating scouting report...
                      </span>
                    </div>
                  ) : report?.report ? (
                    <div className="prose prose-sm dark:prose-invert max-w-none [&_h3]:text-base [&_h3]:font-semibold [&_h3]:mt-4 [&_h3]:mb-2 [&_h2]:text-lg [&_h2]:font-bold [&_h2]:mt-4 [&_h2]:mb-2 [&_p]:mb-2 [&_ul]:mb-2 [&_li]:mb-1 [&_strong]:text-foreground">
                      <ReactMarkdown>{report.report}</ReactMarkdown>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-sm text-muted-foreground">
                      Scouting report will appear here once data is loaded
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
