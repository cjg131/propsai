"use client";

import {
  BarChart3,
  Brain,
  Calculator,
  Database,
  DollarSign,
  FileText,
  Home,
  LineChart,
  Search,
  Settings,
  TrendingUp,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { ThemeToggle } from "@/components/theme-toggle";

const navItems = [
  {
    group: "Dashboard",
    items: [
      { title: "Today's Picks", href: "/", icon: Home },
      { title: "Player Research", href: "/players", icon: Search },
      { title: "Odds Comparison", href: "/odds", icon: TrendingUp },
    ],
  },
  {
    group: "Betting",
    items: [
      { title: "Bet Tracker", href: "/bets", icon: DollarSign },
      { title: "Parlay Builder", href: "/parlays", icon: Calculator },
      { title: "Bankroll", href: "/bankroll", icon: LineChart },
    ],
  },
  {
    group: "Analytics",
    items: [
      { title: "Performance", href: "/performance", icon: BarChart3 },
      { title: "Backtesting", href: "/backtesting", icon: Brain },
      { title: "Paper Trading", href: "/paper-trading", icon: FileText },
    ],
  },
  {
    group: "System",
    items: [
      { title: "Data Management", href: "/data", icon: Database },
      { title: "Settings", href: "/settings", icon: Settings },
    ],
  },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b border-sidebar-border">
        <div className="flex items-center gap-2 px-2 py-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground font-bold text-sm">
            PA
          </div>
          <span className="font-bold text-lg group-data-[collapsible=icon]:hidden">
            PropsAI
          </span>
        </div>
      </SidebarHeader>
      <SidebarContent>
        {navItems.map((group) => (
          <SidebarGroup key={group.group}>
            <SidebarGroupLabel>{group.group}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {group.items.map((item) => (
                  <SidebarMenuItem key={item.href}>
                    <SidebarMenuButton
                      asChild
                      isActive={pathname === item.href}
                      tooltip={item.title}
                    >
                      <Link href={item.href}>
                        <item.icon className="h-4 w-4" />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>
      <SidebarFooter className="border-t border-sidebar-border p-2">
        <div className="flex items-center justify-between group-data-[collapsible=icon]:justify-center">
          <ThemeToggle />
          <span className="text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">
            v0.1.0
          </span>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
