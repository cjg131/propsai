"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CheckCircle2, Settings2, Key, Layers } from "lucide-react";
import {
  useAppSettings,
  useUpdateSettings,
  usePresets,
  type ModelPreset,
} from "@/lib/hooks/use-settings";

export default function SettingsPage() {
  const { data: settings } = useAppSettings();
  const { data: presets } = usePresets();
  const updateMutation = useUpdateSettings();

  const [bankroll, setBankroll] = useState<string | null>(null);
  const [unitSize, setUnitSize] = useState<string | null>(null);
  const [fantasyFormat, setFantasyFormat] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState<string | null>(null);

  const displayBankroll = bankroll ?? String(settings?.bankroll ?? 1000);
  const displayUnitSize = unitSize ?? String(settings?.unit_size ?? 10);
  const displayFantasyFormat = fantasyFormat ?? settings?.fantasy_format ?? "draftkings";
  const displayActivePreset = activePreset ?? settings?.active_preset ?? "balanced";

  const handleSaveGeneral = () => {
    updateMutation.mutate({
      bankroll: parseFloat(displayBankroll),
      unit_size: parseFloat(displayUnitSize),
      fantasy_format: displayFantasyFormat,
    });
  };

  const handleSelectPreset = (presetId: string) => {
    setActivePreset(presetId);
    updateMutation.mutate({ active_preset: presetId });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Configure bankroll, model presets, API keys, and preferences
        </p>
      </div>

      <Tabs defaultValue="general" className="space-y-4">
        <TabsList>
          <TabsTrigger value="general" className="gap-1.5">
            <Settings2 className="h-3.5 w-3.5" />
            General
          </TabsTrigger>
          <TabsTrigger value="presets" className="gap-1.5">
            <Layers className="h-3.5 w-3.5" />
            Presets
          </TabsTrigger>
          <TabsTrigger value="api" className="gap-1.5">
            <Key className="h-3.5 w-3.5" />
            API Keys
          </TabsTrigger>
        </TabsList>

        <TabsContent value="general">
          <Card>
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="bankroll">Bankroll ($)</Label>
                  <Input
                    id="bankroll"
                    type="number"
                    value={displayBankroll}
                    onChange={(e) => setBankroll(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="unit-size">Default Unit Size ($)</Label>
                  <Input
                    id="unit-size"
                    type="number"
                    value={displayUnitSize}
                    onChange={(e) => setUnitSize(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="fantasy-format">Fantasy Scoring Format</Label>
                  <Select value={displayFantasyFormat} onValueChange={setFantasyFormat}>
                    <SelectTrigger id="fantasy-format">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="draftkings">DraftKings</SelectItem>
                      <SelectItem value="fanduel">FanDuel</SelectItem>
                      <SelectItem value="yahoo">Yahoo</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Button onClick={handleSaveGeneral} disabled={updateMutation.isPending}>
                  {updateMutation.isPending ? "Saving..." : "Save Settings"}
                </Button>
                {updateMutation.isSuccess && (
                  <span className="text-sm text-green-500 flex items-center gap-1">
                    <CheckCircle2 className="h-4 w-4" />
                    Saved
                  </span>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="presets">
          <Card>
            <CardHeader>
              <CardTitle>Model Presets</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-3">
                {(presets ?? []).map((preset: ModelPreset) => (
                  <Card
                    key={preset.id}
                    className={`border-2 cursor-pointer transition-colors ${
                      displayActivePreset === preset.id
                        ? "border-primary"
                        : "hover:border-primary/50"
                    }`}
                    onClick={() => handleSelectPreset(preset.id)}
                  >
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center gap-2">
                        {preset.name}
                        {displayActivePreset === preset.id && (
                          <Badge variant="default" className="text-xs">
                            Active
                          </Badge>
                        )}
                        {preset.is_builtin && (
                          <Badge variant="outline" className="text-xs">
                            Built-in
                          </Badge>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">
                        {preset.description}
                      </p>
                      <div className="mt-2 flex gap-3 text-xs text-muted-foreground">
                        <span>Kelly: {(preset.kelly_fraction * 100).toFixed(0)}%</span>
                        <span>Min Conf: {(preset.min_confidence * 100).toFixed(0)}%</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api">
          <Card>
            <CardHeader>
              <CardTitle>API Keys</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                API keys are configured via environment variables on the backend.
                Update your <code className="text-xs bg-muted px-1 py-0.5 rounded">.env</code> file
                in the backend directory.
              </p>
              <div className="space-y-3">
                {[
                  { label: "SportsDataIO", env: "SPORTSDATAIO_API_KEY" },
                  { label: "OpenAI", env: "OPENAI_API_KEY" },
                  { label: "Twitter/X", env: "TWITTER_BEARER_TOKEN" },
                  { label: "Sentry", env: "SENTRY_DSN" },
                  { label: "Supabase URL", env: "SUPABASE_URL" },
                  { label: "Supabase Key", env: "SUPABASE_KEY" },
                ].map((item) => (
                  <div
                    key={item.env}
                    className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                  >
                    <div>
                      <span className="text-sm font-medium">{item.label}</span>
                      <div className="text-xs text-muted-foreground font-mono">
                        {item.env}
                      </div>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      Server-side
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
