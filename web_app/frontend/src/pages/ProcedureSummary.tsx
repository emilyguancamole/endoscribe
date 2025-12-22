import Layout from "@/components/Layout";
import { useEffect, useState } from "react";
import { useLocation, useRoute } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useMutation } from "@tanstack/react-query";
import { processTranscript, ProcessResponse, getSession, saveSessionNote } from "@/lib/api";
import { procedureLabels } from "@/lib/constants";
import Header from "@/components/note/Header";
import PepRiskTab from "@/components/note/PepRiskTab";
import ExtractedFieldsTab from "@/components/note/ExtractedFieldsTab";

export default function ProcedureSummary() {
  const [, params] = useRoute("/procedure-summary/:sessionId");
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  const [sessionId, setSessionId] = useState<string>("");
  const [createdAt, setCreatedAt] = useState<string | null>(null);
  const [procedureType, setProcedureType] = useState<string>("");
  const [transcript, setTranscript] = useState<string>("");
  const [noteContent, setNoteContent] = useState("");
  const [results, setResults] = useState<ProcessResponse | null>(null);
  const [allowAutoProcess, setAllowAutoProcess] = useState<boolean>(false);
  const [sessionLoadFinished, setSessionLoadFinished] = useState<boolean>(false);

  useEffect(() => {
    const sessionIdFromRoute = params?.sessionId;
    const stored = sessionStorage.getItem("currentSession");

    if (stored) {
      const parsed = JSON.parse(stored);
      setSessionId(parsed.sessionId);
      setProcedureType(parsed.procedureType);
      setTranscript(parsed.transcript || "");
      setCreatedAt(parsed.createdAt || parsed.created_at || null);
      setAllowAutoProcess(true);
      setSessionLoadFinished(true);
    } else if (sessionIdFromRoute) {
      setSessionId(sessionIdFromRoute);
      (async () => {
        try {
          const sess = await getSession(sessionIdFromRoute);
          if (sess) {
            setProcedureType(sess.procedure_type || "");
            setTranscript(sess.transcript || "");
            setCreatedAt(sess.created_at || null);
            if (sess.results) {
              setResults(sess.results as ProcessResponse);
              setNoteContent((sess.results.formatted_note as string) || (sess.results.raw_output as string) || "");
            }
          }
        } catch (err) {
          // ignore - leave page to show fallback
          console.error("Failed to load session:", err);
        } finally {
          setSessionLoadFinished(true);
        }
      })();
    } else {
      setLocation("/new-note");
    }
  }, [params?.sessionId, setLocation]);

  // Process transcript
  const processMutation = useMutation({
    mutationFn: processTranscript,
    onSuccess: (data) => {
      console.log("LLM extraction results:", data);
      setResults(data);
      setNoteContent(data.formatted_note || data.raw_output || "");
      toast({
        title: "Processing Complete",
        description: "The transcript has been processed successfully.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Processing Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  // Only auto-process when coming directly from dictation flow
  useEffect(() => {
    if (!allowAutoProcess) return;
    if (!sessionLoadFinished) return;
    if (transcript && procedureType && !results && !processMutation.isPending) {
      processMutation.mutate({
        procedure_type: procedureType,
        transcript: transcript,
        session_id: sessionId,
      });
    }
  }, [allowAutoProcess, sessionLoadFinished, transcript, procedureType, sessionId, results]);

  const handleRegenerate = () => {
    if (!transcript || !procedureType) {
      toast({
        title: "Cannot regenerate",
        description: "Missing transcript or procedure type.",
        variant: "destructive",
      });
      return;
    }
    processMutation.mutate({
      procedure_type: procedureType,
      transcript: transcript,
      session_id: sessionId,
    });
  };

  const handleSave = async () => {
    try {
      const ok = await saveSessionNote(sessionId, { // server saves to backend under results/sessions/
        note_content: noteContent,
        procedure_type: procedureType,
        transcript,
        results: results || undefined,
      });
      if (!ok) throw new Error("Save failed");
      sessionStorage.removeItem("currentSession");
      toast({
        title: "Note Saved",
        description: "Procedure note saved successfully.",
      });
      setTimeout(() => setLocation("/"), 500);
    } catch (e: any) {
      toast({
        title: "Save Failed",
        description: e?.message || "Failed to save note",
        variant: "destructive",
      });
    }
  };

  const isERCP = procedureType === 'ercp';
  const procedureLabel = procedureLabels[procedureType] || procedureType;

  if (!sessionId && !procedureType) {
    return null;
  }

  return (
    <Layout>
      <div className="space-y-6 animate-in fade-in duration-500 pb-20">
        <Header
          procedureLabel={procedureLabel}
          sessionId={sessionId}
          generatedAt={createdAt}
          results={results}
          sessionLoadFinished={sessionLoadFinished}
          handleRegenerate={handleRegenerate}
          handleSave={handleSave}
          processMutation={processMutation}
        />

        {/* Processing State */}
        {processMutation.isPending && (
          <Card className="border-primary/20 bg-primary/5">
            <CardContent className="flex items-center gap-3 py-6">
              <Loader2 className="w-5 h-5 animate-spin text-primary" />
              <span className="text-sm font-medium">Processing transcript with AI...</span>
            </CardContent>
          </Card>
        )}

        {/* If missing saved results, do not auto-run LLM */}
        {!results && sessionLoadFinished && !allowAutoProcess && !processMutation.isPending && (
          <Card className="border-border/50">
            <CardContent className="flex items-center gap-3 py-6">
              <AlertCircle className="w-5 h-5 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground">
                Saved note data not found. Re-generate note to run extraction again.
              </span>
            </CardContent>
          </Card>
        )}

        {/* Error State */}
        {processMutation.isError && (
          <Card className="border-destructive/20 bg-destructive/5">
            <CardContent className="flex items-center gap-3 py-6">
              <AlertCircle className="w-5 h-5 text-destructive" />
              <span className="text-sm font-medium text-destructive">
                Failed to process transcript. Please try again.
              </span>
            </CardContent>
          </Card>
        )}

        <Tabs defaultValue="note" className="space-y-6">
          <TabsList className="bg-white p-1 border border-border/50 h-auto rounded-xl shadow-sm">
            <TabsTrigger value="note" className="py-2.5 px-6 rounded-lg data-[state=active]:bg-primary/5 data-[state=active]:text-primary font-medium">
              Procedure Note
            </TabsTrigger>
            {isERCP && (
              <TabsTrigger value="risk" className="py-2.5 px-6 rounded-lg data-[state=active]:bg-primary/5 data-[state=active]:text-primary font-medium">
                PEP Risk
              </TabsTrigger>
            )}
            <TabsTrigger value="fields" className="py-2.5 px-6 rounded-lg data-[state=active]:bg-primary/5 data-[state=active]:text-primary font-medium">
              Extracted Fields
            </TabsTrigger>
            <TabsTrigger value="transcript" className="py-2.5 px-6 rounded-lg data-[state=active]:bg-primary/5 data-[state=active]:text-primary font-medium">
              Raw Transcript
            </TabsTrigger>
          </TabsList>

          <TabsContent value="note" className="space-y-6">
            <Card className="shadow-sm border-none ring-1 ring-border/50">
              <CardHeader className="bg-muted/20 border-b border-border/50">
                <CardTitle className="text-lg flex justify-between items-center">
                  Clinical Documentation
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <Textarea
                  className="min-h-[500px] resize-y border-0 focus-visible:ring-0 p-6 text-base leading-relaxed font-normal"
                  value={noteContent}
                  onChange={(e) => setNoteContent(e.target.value)}
                  placeholder="Processing transcript..."
                />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="fields">
            <ExtractedFieldsTab results={results} />
          </TabsContent>

          <TabsContent value="transcript">
            <Card className="shadow-sm">
              <CardHeader className="bg-muted/20 border-b border-border/50">
                <CardTitle className="text-lg">Raw Transcript</CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="bg-muted/30 p-4 rounded-lg min-h-[300px] whitespace-pre-wrap text-sm">
                  {transcript || "No transcript available"}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {isERCP && (
            <TabsContent value="risk">
              <PepRiskTab results={results} />
            </TabsContent>
          )}
        </Tabs>
      </div>
    </Layout>
  );
}
