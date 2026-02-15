"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { TrendingUp, BarChart3 } from "lucide-react";

const tools = [
  {
    id: "sportsbook",
    label: "Sportsbook Props",
    href: "/",
    icon: TrendingUp,
    matchPrefix: null, // default — everything not /kalshi
  },
  {
    id: "kalshi",
    label: "Kalshi Predictions",
    href: "/kalshi",
    icon: BarChart3,
    matchPrefix: "/kalshi",
  },
];

export function ToolSwitcher() {
  const pathname = usePathname();
  const activeToolId = pathname.startsWith("/kalshi") ? "kalshi" : "sportsbook";

  return (
    <div className="flex items-center gap-1 rounded-lg bg-muted p-1">
      {tools.map((tool) => (
        <Link
          key={tool.id}
          href={tool.href}
          className={cn(
            "flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
            activeToolId === tool.id
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          <tool.icon className="h-4 w-4" />
          <span className="hidden sm:inline">{tool.label}</span>
        </Link>
      ))}
    </div>
  );
}
