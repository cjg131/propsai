"use client";

import { useKalshiPositions, useKalshiBalance } from "@/lib/hooks/use-kalshi";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Wallet, DollarSign, TrendingUp, AlertCircle } from "lucide-react";

export default function KalshiPositionsPage() {
  const { data: balanceData, error: balanceError } = useKalshiBalance();
  const { data: positionsData, isLoading, error: positionsError } = useKalshiPositions();

  const positions = positionsData?.market_positions || [];
  const hasAuth = !balanceError;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">My Positions</h1>
        <p className="text-sm text-muted-foreground">
          Your current Kalshi portfolio and open positions
        </p>
      </div>

      {!hasAuth ? (
        <Card>
          <CardContent className="p-8 text-center">
            <AlertCircle className="h-12 w-12 text-amber-500 mx-auto mb-3" />
            <p className="font-medium">Kalshi API Key Required</p>
            <p className="text-sm text-muted-foreground mt-1">
              Configure your Kalshi API key in the backend settings to view positions and balance.
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Balance Card */}
          {balanceData && (
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <DollarSign className="h-4 w-4 text-emerald-500" />
                    <span className="text-sm text-muted-foreground">Balance</span>
                  </div>
                  <p className="text-2xl font-bold mt-1">
                    ${((balanceData.balance || 0) / 100).toFixed(2)}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-muted-foreground">Payout</span>
                  </div>
                  <p className="text-2xl font-bold mt-1">
                    ${((balanceData.payout || 0) / 100).toFixed(2)}
                  </p>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Positions */}
          {isLoading ? (
            <Card className="animate-pulse">
              <CardContent className="p-4 h-32" />
            </Card>
          ) : positions.length === 0 ? (
            <Card>
              <CardContent className="p-8 text-center">
                <Wallet className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
                <p className="font-medium">No open positions</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Find edges in the Edge Finder and start trading!
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {positions.map((pos) => (
                <Card key={pos.ticker}>
                  <CardContent className="p-4 flex items-center justify-between">
                    <div>
                      <p className="font-medium text-sm">{pos.market_title || pos.ticker}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline" className="text-[10px]">
                          {pos.side.toUpperCase()}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          Qty: {pos.quantity} @ {pos.average_price}¢
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
