import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ProcessResponse } from "@/lib/api";

export default function ExtractedFieldsTab({ results }: { results: ProcessResponse | null }) {
    return (
        <Card className="shadow-sm">
            <CardHeader className="bg-muted/20 border-b border-border/50">
                <CardTitle className="text-lg flex justify-between items-center">
                    LLM-Extracted Fields (Debug View)
                </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
                {results ? (
                    <div className="space-y-4">
                        {/* colonoscopy data */}
                        {results.colonoscopy_data && (
                            <div className="space-y-2">
                                <h3 className="font-semibold text-primary">Colonoscopy Data</h3>
                                <pre className="bg-muted/50 p-4 rounded-lg overflow-auto text-xs">
                                    {JSON.stringify(results.colonoscopy_data, null, 2)}
                                </pre>
                            </div>
                        )}

                        {/* polyps data */}
                        {results.polyps_data && results.polyps_data.length > 0 && (
                            <div className="space-y-2">
                                <h3 className="font-semibold text-primary">Polyps Data</h3>
                                <pre className="bg-muted/50 p-4 rounded-lg overflow-auto text-xs">
                                    {JSON.stringify(results.polyps_data, null, 2)}
                                </pre>
                            </div>
                        )}

                        {/* procedure data */}
                        {results.procedure_data && (
                            <div className="space-y-2">
                                <h3 className="font-semibold text-primary">Procedure Data</h3>
                                <pre className="bg-muted/50 p-4 rounded-lg overflow-auto text-xs">
                                    {JSON.stringify(results.procedure_data, null, 2)}
                                </pre>
                            </div>
                        )}

                        {/* PEP risk data */}
                        {results.pep_risk_data && (
                            <div className="space-y-2">
                                <h3 className="font-semibold text-primary">PEP Risk Data</h3>
                                <pre className="bg-muted/50 p-4 rounded-lg overflow-auto text-xs">
                                    {JSON.stringify(results.pep_risk_data, null, 2)}
                                </pre>
                                {typeof results.pep_risk_score === 'number' && (
                                    <div className="mt-2 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                                        <p className="text-sm font-medium">
                                            PEP Risk Score: <span className="text-amber-700">{results.pep_risk_score.toFixed(2)}%</span>
                                        </p>
                                        {results.pep_risk_category && (
                                            <p className="text-sm">
                                                Category: <span className="font-medium">{results.pep_risk_category}</span>
                                            </p>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ) : (
                    <p className="text-muted-foreground">Processing data...</p>
                )}
            </CardContent>
        </Card>
    )
}