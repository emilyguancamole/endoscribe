import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { Mic, History, Settings, Activity, AlertCircle, CheckCircle2 } from "lucide-react";
import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const [location] = useLocation();
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  // Health check query
  const { data: healthData } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const res = await fetch('/health');
      if (!res.ok) throw new Error('Health check failed');
      return res.json();
    },
    refetchInterval: 60000,
    retry: 3,
  });

  const healthStatus = healthData?.status || 'unknown';
  const isHealthy = healthStatus === 'healthy';

  const navItems = [
    { href: "/new-note", label: "New Note", icon: Mic },
    { href: "/", label: "History", icon: History },
  ];

  return (
    <div className="min-h-screen bg-background flex flex-col md:flex-row font-sans">
      {/* Mobile Header */}
      <div className="md:hidden flex items-center justify-between p-4 bg-white border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-md bg-primary flex items-center justify-center text-white font-bold text-sm">
            ES
          </div>
          <span className="font-bold text-lg text-primary tracking-tight">EndoScribe</span>
        </div>
        <button
          onClick={() => setIsMobileOpen(!isMobileOpen)}
          className="p-2 text-primary"
          aria-label="Toggle menu"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>

      {/* Sidebar */}
      <aside className={cn(
        "fixed md:sticky top-0 z-40 h-screen w-64 bg-white border-r border-border flex-col transition-transform duration-300 ease-in-out md:translate-x-0 md:flex shadow-sm",
        isMobileOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="p-6 border-b border-border/40">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-lg shadow-sm bg-primary flex items-center justify-center text-white font-bold">
              ES
            </div>
            <h1 className="font-bold text-xl text-primary tracking-tight">EndoScribe</h1>
          </div>
          <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider pl-1">
            Medical Documentation
          </p>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location === item.href || (item.href !== "/" && location.startsWith(item.href));

            return (
              <Link key={item.href} href={item.href}>
                <a
                  className={cn(
                    "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 group",
                    isActive
                      ? "bg-primary/5 text-primary shadow-sm border border-primary/10"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                  onClick={() => setIsMobileOpen(false)}
                >
                  <Icon className={cn("w-5 h-5", isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground")} />
                  {item.label}
                </a>
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-border/40 space-y-2">
          {/* System Health Status */}
          <div className="px-4 py-3 rounded-lg bg-muted/30">
            <div className="flex items-center gap-2 mb-1">
              <Activity className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs font-medium text-muted-foreground">System Status</span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              {isHealthy ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-green-600" />
                  <span className="text-xs font-medium text-green-700">All Systems Operational</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-4 h-4 text-amber-600" />
                  <span className="text-xs font-medium text-amber-700">
                    {healthStatus === 'degraded' ? 'Limited Functionality' : 'Checking...'}
                  </span>
                </>
              )}
            </div>
            {healthData?.transcription_service && (
              <p className="text-[10px] text-muted-foreground mt-1">
                Transcription: {healthData.transcription_service}
              </p>
            )}
          </div>

          <button className="flex items-center gap-3 px-4 py-3 w-full rounded-lg text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors">
            <Settings className="w-5 h-5" />
            Settings
          </button>

          <div className="pt-2 px-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-secondary/20 flex items-center justify-center text-secondary font-bold text-xs">
                MD
              </div>
              <div className="flex-1 overflow-hidden">
                <p className="text-sm font-medium truncate text-foreground">Medical Provider</p>
                <p className="text-xs text-muted-foreground truncate">Gastroenterology</p>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto bg-background/50 relative">
        <div className="absolute inset-0 bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_70%,transparent_100%)] pointer-events-none opacity-50"></div>
        <div className="relative p-4 md:p-8 max-w-7xl mx-auto min-h-[calc(100vh-4rem)]">
          {children}
        </div>
      </main>

      {/* Mobile Overlay */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={() => setIsMobileOpen(false)}
        />
      )}
    </div>
  );
}
