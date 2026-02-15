"use client";

import {
  BarChart3,
  Bot,
  Crosshair,
  History,
  Wallet,
  Settings,
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

const kalshiNavItems = [
  {
    group: "Markets",
    items: [
      { title: "Today's Markets", href: "/kalshi", icon: BarChart3 },
      { title: "Edge Finder", href: "/kalshi/edges", icon: Crosshair },
      { title: "History", href: "/kalshi/history", icon: History },
      { title: "Agent", href: "/kalshi/agent", icon: Bot },
    ],
  },
  {
    group: "Account",
    items: [
      { title: "My Positions", href: "/kalshi/positions", icon: Wallet },
      { title: "Settings", href: "/kalshi/settings", icon: Settings },
    ],
  },
];

export function KalshiSidebar() {
  const pathname = usePathname();

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b border-sidebar-border">
        <div className="flex items-center gap-2 px-2 py-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-600 text-white font-bold text-sm">
            K
          </div>
          <span className="font-bold text-lg group-data-[collapsible=icon]:hidden">
            Kalshi
          </span>
        </div>
      </SidebarHeader>
      <SidebarContent>
        {kalshiNavItems.map((group) => (
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
