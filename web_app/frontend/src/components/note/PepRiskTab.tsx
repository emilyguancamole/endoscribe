import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ProcessResponse } from "@/lib/api";

interface TreatmentPrediction {
    therapy_id?: string;
    therapy_label?: string;
    risk_percentage: number;
    risk_category?: string;
}

// interface PepRiskTabProps {
//     results: {
//         pep_risk_score?: number;
//         pep_risk_category?: string;
//         treatment_predictions?: TreatmentPrediction[];
//     } | null;
// }

export default function PepRiskTab({ results }: { results: ProcessResponse | null }) {
    return (
        <Card className="shadow-sm">
            <CardHeader className="bg-muted/20 border-b border-border/50">
                <CardTitle className="text-lg flex justify-between items-center">
                    Post-ERCP Pancreatitis (PEP) Risk Assessment
                </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
                {typeof results?.pep_risk_score === 'number' ? (
                    <div className="space-y-4">
                        <div className="text-center space-y-3">
                            <div className="text-lg">Estimated risk without prophylaxis:</div>
                            <div className="text-5xl font-bold">
                                {results.pep_risk_score.toFixed(1)}%
                            </div>
                            {results.treatment_predictions && (
                                <div className="mt-8 flex flex-col items-center gap-2">
                                    <div className="text-lg">Estimated risk under prophylaxis:</div>
                                    <div className="flex flex-wrap gap-2 justify-center mt-1">
                                        {results.treatment_predictions.map((tp: TreatmentPrediction, i: number) => {
                                            const label = tp.therapy_label || tp.therapy_id || "Unknown";
                                            if ((label || "").toLowerCase() === "no treatment") return null;
                                            return (
                                                <div key={i} className="border rounded-md px-4 py-2 flex flex-col items-center min-w-[120px]">
                                                    <div className="text-xl">{(tp.risk_percentage).toFixed(1)}%</div>
                                                    <div className="font-medium">{label}</div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="bg-amber-50 mt-12 p-4 rounded-lg">
                            <p className="text-sm text-amber-900">
                                This risk assessment is based on risk factors extracted from the transcript by EndoScribe.
                                Always use clinical judgment in conjunction with these automated assessments.
                            </p>
                        </div>
                    </div>
                ) : (
                    <p className="text-muted-foreground">Not available.</p>
                )}
            </CardContent>
        </Card>
    )
}