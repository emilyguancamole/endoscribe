function DataTable({ data }) {
  if (!data || Object.keys(data).length === 0) {
    return <p className="text-base-content/70">No data available</p>;
  }

  return (
    <table className="table table-zebra w-full">
      <tbody>
        {Object.entries(data).map(([key, value]) => {
          const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
          let displayValue = value;

          if (Array.isArray(value)) {
            displayValue = (
              <ul className="list-disc list-inside">
                {value.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            );
          } else if (typeof value === 'object' && value !== null) {
            displayValue = <pre className="whitespace-pre-wrap">{JSON.stringify(value, null, 2)}</pre>;
          } else {
            displayValue = String(value);
          }

          return (
            <tr key={key}>
              <th className="w-1/3">{displayKey}</th>
              <td>{displayValue}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function PolypsTable({ polyps }) {
  if (!polyps || polyps.length === 0) {
    return <p className="text-base-content/70">No polyps detected</p>;
  }

  const headers = Object.keys(polyps[0] || {}).filter(k => k !== 'participant_id' && k !== 'polyp_id');

  return (
    <table className="table table-compact w-full">
      <thead>
        <tr>
          <th>Polyp #</th>
          {headers.map(header => (
            <th key={header}>
              {header.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {polyps.map((polyp, idx) => (
          <tr key={idx}>
            <td>{idx + 1}</td>
            {headers.map(header => (
              <td key={header}>{String(polyp[header])}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ResultsDisplay({
  procedureType,
  colonoscopyData,
  polypsData,
  procedureData,
  pepRiskData,
  pepRiskScore,
  pepRiskCategory,
  showResults,
  showPEPRiskResults
}) {
  if (!showResults && !showPEPRiskResults) return null;

  const renderPEPRiskPrediction = () => {
    if (!pepRiskData?.prediction) return null;

    const { prediction } = pepRiskData;
    const score = typeof pepRiskScore === 'number' ? pepRiskScore : prediction.risk_score;
    const category = pepRiskCategory || prediction.risk_category;
    const treatmentPredictions = prediction.treatment_predictions || [];

    const categoryColors = {
      low: 'bg-green-100 text-green-800 border-green-300',
      moderate: 'bg-yellow-100 text-yellow-800 border-yellow-300',
      high: 'bg-red-100 text-red-800 border-red-300'
    };

    const colorClass = categoryColors[category] || 'bg-gray-100 text-gray-800 border-gray-300';

    const noTreatmentPrediction = treatmentPredictions.find(tp =>
      tp.therapy?.toLowerCase().includes('no') || tp.therapy?.toLowerCase().includes('baseline')
    );

    const treatmentOptions = treatmentPredictions.filter(tp =>
      !(tp.therapy?.toLowerCase().includes('no') || tp.therapy?.toLowerCase().includes('baseline'))
    );

    const displayScore = noTreatmentPrediction?.risk_percentage ?? score;

    return (
      <div className="mb-6">
        <div className={`p-6 rounded-lg border-2 ${colorClass}`}>
          <h3 className="text-2xl font-bold mb-2">Predicted PEP Risk</h3>

          <div className="flex items-baseline gap-4 mb-4">
            <div className="text-5xl font-bold">{Number(displayScore).toFixed(1)}%</div>
            <div className="text-l font-semibold">without prophylaxis</div>
          </div>

          {treatmentOptions.length > 0 && (
            <div className="mt-6 pt-4 border-t border-gray-300">
              <h4 className="font-semibold mb-3 text-base">With Prophylaxis:</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {treatmentOptions.map((tp, idx) => {
                  const tpColorClass = categoryColors[tp.risk_category] || 'bg-gray-50';
                  return (
                    <div key={idx} className={`p-3 rounded border ${tpColorClass.replace('100', '50')}`}>
                      <div className="flex justify-between items-center">
                        <div className="text-sm font-medium">{tp.therapy}</div>
                        <div className="text-lg font-bold">{tp.risk_percentage}%</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {prediction.top_risk_factors && prediction.top_risk_factors.length > 0 && (
            <div className="mt-4">
              <h4 className="font-semibold mb-2">Top Contributing Factors:</h4>
              <ul className="list-disc list-inside">
                {prediction.top_risk_factors.map(([factor, contribution], idx) => (
                  <li key={idx} className="text-sm">
                    {factor.replace(/_/g, ' ')}: {contribution > 0 ? '+' : ''}{Number(contribution).toFixed(1)} points
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {showPEPRiskResults && renderPEPRiskPrediction()}

      {showResults && (
        <div>
          {procedureType === 'col' ? (
            <>
              <h3 className="text-xl font-semibold mb-2">Colonoscopy Findings</h3>
              <DataTable data={colonoscopyData} />

              <h3 className="text-xl font-semibold mt-4 mb-2">Polyps</h3>
              <PolypsTable polyps={polypsData} />
            </>
          ) : (
            <>
              <h3 className="text-xl font-semibold mb-2">Procedure Data</h3>
              <DataTable data={procedureData} />
            </>
          )}
        </div>
      )}

      {!showResults && !showPEPRiskResults && (
        <p className="text-base-content/70">No results to display</p>
      )}
    </div>
  );
}
