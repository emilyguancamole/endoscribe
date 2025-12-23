import { Button } from "@/components/ui/button";
import { Save, Printer, ArrowLeft, } from "lucide-react";
import { ProcessResponse } from "@/lib/api";
import { useLocation } from "wouter";

interface HeaderProps {
    procedureLabel: string;
    sessionId: string;
    generatedAt?: string | null;
    results: ProcessResponse | null;
    sessionLoadFinished: boolean;
    handleRegenerate: () => void;
    handleSave: () => void;
    processMutation: {
        isPending: boolean;
    };
}
export default function Header({
    procedureLabel,
    sessionId,
    generatedAt,
    results,
    sessionLoadFinished,
    handleRegenerate,
    handleSave,
    processMutation,
}: HeaderProps) {
    const [, setLocation] = useLocation();
    return (
        <div className="flex flex-col gap-4">
            <Button
                variant="ghost"
                onClick={() => setLocation("/")}
                className="pl-0 hover:bg-transparent hover:text-primary self-start"
            >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to History
            </Button>

            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border/50 pb-6">
                <div>
                    <div className="flex items-center gap-3 mb-1">
                        <h1 className="text-3xl font-bold text-primary">Procedure Summary</h1>
                        <span className="px-3 py-1 bg-amber-100 text-amber-700 rounded-full text-xs font-bold uppercase tracking-wide">
                            Draft
                        </span>
                    </div>
                    <p className="text-muted-foreground">
                        {procedureLabel} • {generatedAt ? new Date(generatedAt).toLocaleString() : `Session ${sessionId.slice(0, 8)}`}
                    </p>
                </div>
                <div className="flex gap-2">
                    {/* <Button variant="outline" className="bg-white">
                        <Printer className="w-4 h-4 mr-2" />
                        Print
                    </Button> */}
                    {!results && sessionLoadFinished && (
                        <Button variant="outline" className="bg-white" onClick={handleRegenerate} disabled={processMutation.isPending}>
                            Re-generate Note
                        </Button>
                    )}
                    <Button className="bg-primary shadow-lg shadow-primary/20" onClick={handleSave}>
                        <Save className="w-4 h-4 mr-2" />
                        Save Note
                    </Button>
                </div>
            </div>
        </div>
    )
}