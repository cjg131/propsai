"use client";

import { usePathname } from "next/navigation";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { KalshiSidebar } from "@/components/kalshi-sidebar";
import { ToolSwitcher } from "@/components/tool-switcher";

export function LayoutShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isKalshi = pathname.startsWith("/kalshi");

  return (
    <SidebarProvider>
      {isKalshi ? <KalshiSidebar /> : <AppSidebar />}
      <SidebarInset>
        <header className="flex h-14 shrink-0 items-center gap-4 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <ToolSwitcher />
        </header>
        <main className="flex-1 overflow-auto p-4 md:p-6">
          {children}
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
