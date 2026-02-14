"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Database,
  RefreshCw,
  Brain,
  Download,
  Users,
  Calendar,
  Layers,
  Activity,
  CheckCircle2,
  Clock,
  Target,
  Camera,
  DollarSign,
  Zap,
  XCircle,
} from "lucide-react";
import {
  useDataStatus,
  useRefreshData,
  useRetrainModels,
  useExportData,
  useEvaluatePredictions,
  useSnapshotLines,
  useRefreshOdds,
  usePrefetchData,
} from "@/lib/hooks/use-data";

interface TaskState {
  isRunning: boolean;
  progress: number;
  label: string;
}

function useSimulatedProgress(isPending: boolean, durationMs: number = 8000) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!isPending) {
      if (progress > 0) setProgress(100);
      return;
    }
    setProgress(5);
    const step = 100 / (durationMs / 200);
    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= 92) return p;
        return Math.min(p + step * (1 - p / 100), 92);
      });
    }, 200);
    return () => clearInterval(interval);
  }, [isPending, durationMs]);

  useEffect(() => {
    if (!isPending) {
      const t = setTimeout(() => setProgress(0), 2000);
      return () => clearTimeout(t);
    }
  }, [isPending]);

  return progress;
}

function ActionCard({
  icon: Icon,
  title,
  description,
  buttonLabel,
  isPending,
  isSuccess,
  isError,
  progress,
  onClick,
  successMessage,
  resultData,
  variant = "default",
}: {
  icon: any;
  title: string;
  description: string;
  buttonLabel: string;
  isPending: boolean;
  isSuccess: boolean;
  isError: boolean;
  progress: number;
  onClick: () => void;
  successMessage?: string;
  resultData?: any;
  variant?: "default" | "destructive" | "outline";
}) {
  return (
    <Card className="relative overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Icon className={`h-4 w-4 ${isPending ? "animate-spin" : ""}`} />
          {title}
          {isPending && (
            <Badge variant="outline" className="text-[10px] ml-auto animate-pulse">
              Running
            </Badge>
          )}
          {isSuccess && !isPending && (
            <CheckCircle2 className="h-3.5 w-3.5 text-green-500 ml-auto" />
          )}
          {isError && !isPending && (
            <XCircle className="h-3.5 w-3.5 text-red-500 ml-auto" />
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">{description}</p>

        {isPending && (
          <div className="space-y-1">
            <Progress value={progress} className="h-1.5" />
            <p className="text-[10px] text-muted-foreground text-right">
              {Math.round(progress)}%
            </p>
          </div>
        )}

        {isSuccess && !isPending && successMessage && (
          <div className="p-2 rounded bg-green-500/10 border border-green-500/20">
            <p className="text-[11px] text-green-600 dark:text-green-400">
              {successMessage}
            </p>
          </div>
        )}

        {isError && !isPending && (
          <div className="p-2 rounded bg-red-500/10 border border-red-500/20">
            <p className="text-[11px] text-red-600 dark:text-red-400">
              Failed — check backend logs
            </p>
          </div>
        )}

        <Button
          size="sm"
          variant={variant === "default" ? "default" : "outline"}
          className="w-full"
          onClick={onClick}
          disabled={isPending}
        >
          {isPending ? (
            <>
              <RefreshCw className="mr-2 h-3 w-3 animate-spin" />
              Running...
            </>
          ) : (
            buttonLabel
          )}
        </Button>
      </CardContent>
    </Card>
  );
}

export default function DataPage() {
  const { data: status, isLoading } = useDataStatus();
  const refreshMutation = useRefreshData();
  const retrainMutation = useRetrainModels();
  const exportMutation = useExportData();
  const evaluateMutation = useEvaluatePredictions();
  const snapshotMutation = useSnapshotLines();
  const oddsMutation = useRefreshOdds();
  const prefetchMutation = usePrefetchData();

  const refreshProgress = useSimulatedProgress(refreshMutation.isPending, 10000);
  const retrainProgress = useSimulatedProgress(retrainMutation.isPending, 30000);
  const evaluateProgress = useSimulatedProgress(evaluateMutation.isPending, 8000);
  const snapshotProgress = useSimulatedProgress(snapshotMutation.isPending, 5000);
  const oddsProgress = useSimulatedProgress(oddsMutation.isPending, 12000);
  const prefetchProgress = useSimulatedProgress(prefetchMutation.isPending, 60000);

  const quotaPct =
    status && status.api_quota_limit > 0
      ? (status.api_quota_used / status.api_quota_limit) * 100
      : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Management</h1>
        <p className="text-muted-foreground">
          Trigger data operations, retrain models, and manage your pipeline
        </p>
      </div>

      {/* Primary Actions */}
      <div>
        <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <Zap className="h-4 w-4" />
          Quick Actions
        </h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <ActionCard
            icon={RefreshCw}
            title="Refresh Data"
            description="Pull latest game results, stats, and injury reports from all APIs"
            buttonLabel="Refresh Now"
            isPending={refreshMutation.isPending}
            isSuccess={refreshMutation.isSuccess}
            isError={refreshMutation.isError}
            progress={refreshProgress}
            onClick={() => refreshMutation.mutate()}
            successMessage="Data refreshed successfully"
          />
          <ActionCard
            icon={DollarSign}
            title="Refresh Odds"
            description="Fetch latest sportsbook odds from The Odds API for all games"
            buttonLabel="Refresh Odds"
            isPending={oddsMutation.isPending}
            isSuccess={oddsMutation.isSuccess}
            isError={oddsMutation.isError}
            progress={oddsProgress}
            onClick={() => oddsMutation.mutate()}
            successMessage={
              oddsMutation.data
                ? `Loaded ${(oddsMutation.data as any)?.lines_stored ?? "?"} prop lines`
                : "Odds refreshed"
            }
          />
          <ActionCard
            icon={Target}
            title="Evaluate Predictions"
            description="Compare yesterday&apos;s predictions against actual game results"
            buttonLabel="Evaluate Now"
            isPending={evaluateMutation.isPending}
            isSuccess={evaluateMutation.isSuccess}
            isError={evaluateMutation.isError}
            progress={evaluateProgress}
            onClick={() => evaluateMutation.mutate(undefined)}
            successMessage={
              evaluateMutation.data
                ? `Evaluated ${(evaluateMutation.data as any)?.evaluated ?? "?"} predictions`
                : "Evaluation complete"
            }
          />
          <ActionCard
            icon={Camera}
            title="Snapshot Lines"
            description="Store timestamped line snapshots for Closing Line Value tracking"
            buttonLabel="Snapshot Now"
            isPending={snapshotMutation.isPending}
            isSuccess={snapshotMutation.isSuccess}
            isError={snapshotMutation.isError}
            progress={snapshotProgress}
            onClick={() => snapshotMutation.mutate()}
            successMessage={
              snapshotMutation.data
                ? `Stored ${(snapshotMutation.data as any)?.snapshots_stored ?? "?"} snapshots`
                : "Lines snapshotted"
            }
          />
        </div>
      </div>

      {/* Model & Data Pipeline */}
      <div>
        <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <Brain className="h-4 w-4" />
          Model & Pipeline
        </h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <ActionCard
            icon={Brain}
            title="Retrain Models"
            description="Retrain SmartPredictor ensemble (XGBoost + RF + GB + Ridge meta) on all BDL data"
            buttonLabel="Retrain Now"
            isPending={retrainMutation.isPending}
            isSuccess={retrainMutation.isSuccess}
            isError={retrainMutation.isError}
            progress={retrainProgress}
            onClick={() => retrainMutation.mutate()}
            successMessage="All models retrained successfully"
          />
          <ActionCard
            icon={Database}
            title="Prefetch BDL Data"
            description="Background fetch of BallDontLie game logs to local cache (slow, ~5min)"
            buttonLabel="Start Prefetch"
            isPending={prefetchMutation.isPending}
            isSuccess={prefetchMutation.isSuccess}
            isError={prefetchMutation.isError}
            progress={prefetchProgress}
            onClick={() => prefetchMutation.mutate()}
            successMessage="BDL data prefetch started in background"
          />
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Download className="h-4 w-4" />
                Export Data
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Export predictions, bets, and stats as CSV or JSON
              </p>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  className="flex-1"
                  onClick={() => exportMutation.mutate("csv")}
                  disabled={exportMutation.isPending}
                >
                  CSV
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="flex-1"
                  onClick={() => exportMutation.mutate("json")}
                  disabled={exportMutation.isPending}
                >
                  JSON
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Players</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? "..." : (status?.total_players ?? 0).toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">Active NBA players</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Games</CardTitle>
            <Calendar className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? "..." : (status?.total_games ?? 0).toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">Historical games loaded</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Seasons</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? "..." : (status?.total_seasons ?? 0)}
            </div>
            <p className="text-xs text-muted-foreground">Seasons of data</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Status</CardTitle>
            {status?.model_last_trained ? (
              <CheckCircle2 className="h-4 w-4 text-green-500" />
            ) : (
              <Clock className="h-4 w-4 text-muted-foreground" />
            )}
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {status?.model_last_trained ? "Trained" : "Not Trained"}
            </div>
            <p className="text-xs text-muted-foreground">
              {status?.model_last_trained
                ? `Last: ${new Date(status.model_last_trained).toLocaleDateString()}`
                : "Run retrain to initialize models"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* API Quota */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Activity className="h-4 w-4" />
            API Usage
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">SportsDataIO Daily Quota</span>
            <span className="font-mono">
              {status?.api_quota_used ?? 0} / {status?.api_quota_limit ?? 0}
            </span>
          </div>
          <Progress value={quotaPct} className="h-2" />
          <div className="text-xs text-muted-foreground">
            {quotaPct < 80
              ? "Quota usage is healthy"
              : quotaPct < 95
              ? "Approaching daily limit"
              : "Daily limit nearly reached"}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
