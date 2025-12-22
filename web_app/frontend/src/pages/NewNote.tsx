import Layout from "@/components/Layout";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { procedureTypes, procedureLabels } from "@/lib/constants";
import DictationInterface from "@/components/DictationInterface";
import { ChevronRight, ArrowLeft, Check } from "lucide-react";
import { useLocation } from "wouter";

export default function NewNote() {
  const [step, setStep] = useState<1 | 2>(1); // 1: procedure type, 2: dictation
  const [selectedType, setSelectedType] = useState<string>("");
  const [, setLocation] = useLocation();

  const handleDictationComplete = (text: string, sessionId: string) => {
    sessionStorage.setItem("currentSession", JSON.stringify({
      sessionId,
      procedureType: selectedType,
      transcript: text,
      timestamp: new Date().toISOString()
    }));

    setLocation(`/procedure-summary/${sessionId}`);
  };

  const currentProcedureLabel = procedureLabels[selectedType] || selectedType;

  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        {/* Progress Stepper */}
        <div className="mb-12">
          <div className="flex items-center justify-between relative">
            <div className="absolute left-0 right-0 top-1/2 h-0.5 bg-border -z-10" />

            {[1, 2].map((s) => (
              <div key={s} className="flex flex-col items-center gap-2 bg-background px-2">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all duration-300 ${step >= s
                  ? "bg-primary text-white shadow-lg shadow-primary/20 scale-110"
                  : "bg-muted text-muted-foreground border-2 border-border"
                  }`}>
                  {step > s ? <Check className="w-5 h-5" /> : s}
                </div>
                <span className={`text-xs font-medium ${step >= s ? "text-primary" : "text-muted-foreground"}`}>
                  {s === 1 ? "Procedure Type" : "Dictation"}
                </span>
              </div>
            ))}
          </div>
        </div>

        {step === 1 && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="text-center space-y-2">
              <h1 className="text-3xl font-bold text-primary">Start New Note</h1>
              <p className="text-muted-foreground">Select the procedure type to begin dictation.</p>
            </div>

            <Card className="p-8 border-none shadow-lg bg-white/50 backdrop-blur-sm">
              <div className="grid gap-6 max-w-md mx-auto">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Procedure Type</label>
                  <Select onValueChange={setSelectedType} value={selectedType}>
                    <SelectTrigger className="h-12 text-base">
                      <SelectValue placeholder="Select Procedure Type" />
                    </SelectTrigger>
                    <SelectContent>
                      {procedureTypes[0].subtypes.map(subtype => (
                        <SelectItem key={subtype} value={subtype}>
                          {procedureLabels[subtype]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="pt-4">
                  <Button
                    className="w-full h-12 text-lg shadow-lg shadow-primary/20"
                    disabled={!selectedType}
                    onClick={() => setStep(2)}
                  >
                    Continue to Dictation
                    <ChevronRight className="w-5 h-5 ml-2" />
                  </Button>
                </div>
              </div>
            </Card>
          </div>
        )}

        {step === 2 && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="mb-6">
              <Button variant="ghost" onClick={() => setStep(1)} className="pl-0 hover:bg-transparent hover:text-primary">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Procedure Selection
              </Button>
            </div>

            <DictationInterface
              procedureType={currentProcedureLabel}
              onComplete={handleDictationComplete}
            />
          </div>
        )}
      </div>
    </Layout>
  );
}
