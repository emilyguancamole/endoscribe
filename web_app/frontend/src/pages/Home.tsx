import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Plus, Search, Filter, Calendar, ChevronRight, FileCheck, FileClock } from "lucide-react";
import { Link } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { listSessions } from "@/lib/api";
import { procedureLabels } from "@/lib/constants";

export default function Home() {
  const { data: sessions = [], isLoading } = useQuery({
    queryKey: ['sessions'],
    queryFn: listSessions,
    refetchInterval: 10000,
  });

  return (
    <Layout>
      <div className="space-y-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-primary mb-2">Welcome back</h1>
            <p className="text-muted-foreground">
              You have <span className="text-foreground font-medium">{sessions.filter(s => !s.processed).length} notes</span> pending review.
            </p>
          </div>
          <Link href="/new-note">
            <Button size="lg" className="bg-primary hover:bg-primary/90 text-white shadow-lg shadow-primary/20 transition-all hover:scale-105">
              <Plus className="w-5 h-5 mr-2" />
              New Procedure Note
            </Button>
          </Link>
        </div>

        {/* Filters & Search */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search sessions, procedures..."
              className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-input bg-white focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all"
            />
          </div>
          <Button variant="outline" className="bg-white">
            <Filter className="w-4 h-4 mr-2" />
            Filter
          </Button>
          <Button variant="outline" className="bg-white">
            <Calendar className="w-4 h-4 mr-2" />
            Date Range
          </Button>
        </div>

        {/* History Grid */}
        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">
            Loading sessions...
          </div>
        ) : sessions.length === 0 ? (
          <div className="text-center py-12">
            <FileCheck className="w-16 h-16 mx-auto text-muted-foreground/50 mb-4" />
            <p className="text-muted-foreground">No procedure notes yet. Create your first one!</p>
          </div>
        ) : (
          <div className="grid gap-4">
            {sessions.map((session) => (
              <Link key={session.session_id} href={`/procedure-summary/${session.session_id}`}>
                <Card className="group hover:shadow-md transition-all duration-200 border-l-4 border-l-transparent hover:border-l-primary cursor-pointer">
                  <div className="p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex items-start gap-4">
                      <div className="w-10 h-10 rounded-full bg-primary/5 flex items-center justify-center text-primary mt-1">
                        {session.processed ? <FileCheck className="w-5 h-5" /> : <FileClock className="w-5 h-5" />}
                      </div>
                      <div>
                        <h3 className="font-semibold text-lg text-foreground group-hover:text-primary transition-colors">
                          {procedureLabels[session.procedure_type] || session.procedure_type}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          Session {session.session_id.slice(0, 8)} • {new Date(session.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 justify-between md:justify-end w-full md:w-auto">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${session.processed
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-100'
                        : 'bg-amber-50 text-amber-700 border-amber-100'
                        }`}>
                        {session.processed ? 'Completed' : 'Draft'}
                      </span>
                      <Button variant="ghost" size="icon" className="text-muted-foreground group-hover:text-primary">
                        <ChevronRight className="w-5 h-5" />
                      </Button>
                    </div>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </div>
    </Layout>
  );
}
