import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface PepRiskTabProps {
    results: {
        pep_risk_score?: number;
        pep_risk_category?: string;
    } | null;
}

export default function PepRiskTab({ results }: PepRiskTabProps) {
    console.log("PEP Risk Results:", results);
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
                        <div className="text-center">
                            <div className="text-5xl font-bold text-amber-600">
                                {results.pep_risk_score.toFixed(1)}%
                            </div>
                            <p className="text-sm text-muted-foreground mt-2">
                                Risk Level: <span className="font-semibold">{results.pep_risk_category}</span>
                            </p>
                        </div>
                        <div className="bg-amber-50 p-4 rounded-lg">
                            <p className="text-sm text-amber-900">
                                This risk assessment is based on patient and procedure characteristics extracted from the transcript.
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