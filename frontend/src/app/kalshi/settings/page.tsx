"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Settings, Key, Shield, ExternalLink } from "lucide-react";

export default function KalshiSettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Kalshi Settings</h1>
        <p className="text-sm text-muted-foreground">
          Configure your Kalshi API connection
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Key className="h-4 w-4" />
            API Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">API Key Status</p>
              <p className="text-xs text-muted-foreground">
                RSA key-pair authentication
              </p>
            </div>
            <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30">
              Configured
            </Badge>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Access Level</p>
              <p className="text-xs text-muted-foreground">
                Read/Write (market data + trading)
              </p>
            </div>
            <Badge variant="outline" className="bg-blue-500/10 text-blue-600 border-blue-500/30">
              Read/Write
            </Badge>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Environment</p>
              <p className="text-xs text-muted-foreground">
                Production API
              </p>
            </div>
            <Badge variant="outline">Production</Badge>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Shield className="h-4 w-4" />
            Security
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Your Kalshi private key is stored locally on the backend server and never exposed to the frontend.
            All authenticated requests are signed server-side using RSA-PSS.
          </p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Manage API keys:</span>
            <a
              href="https://kalshi.com/account/profile"
              target="_blank"
              rel="noopener noreferrer"
              className="text-emerald-600 hover:underline flex items-center gap-1"
            >
              Kalshi Profile Settings
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-muted/50">
        <CardContent className="p-4">
          <h3 className="text-sm font-semibold mb-1 flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Coming Soon
          </h3>
          <ul className="text-xs text-muted-foreground space-y-1 mt-2">
            <li>• Auto-trading: automatically place orders when edge signals fire</li>
            <li>• Position sizing: Kelly criterion for Kalshi contracts</li>
            <li>• P&L tracking: historical performance on Kalshi trades</li>
            <li>• Notifications: alerts when new edges are detected</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
