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
            displayValue = JSON.stringify(value, null, 2);
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
              <td key={header}>{polyp[header]}</td>
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
  if (!showResults && !showPEPRiskResults) {
    return null;
  }

  // Helper to render PEP risk prediction
  const renderPEPRiskPrediction = () => {
    if (!pepRiskData?.prediction) return null;

    const { prediction } = pepRiskData;
    const score = pepRiskScore || prediction.risk_score;
    const category = pepRiskCategory || prediction.risk_category;

    // Color coding for risk categories
    const categoryColors = {
      low: 'bg-green-100 text-green-800 border-green-300',
      moderate: 'bg-yellow-100 text-yellow-800 border-yellow-300',
      high: 'bg-red-100 text-red-800 border-red-300'
    };

    const colorClass = categoryColors[category] || 'bg-gray-100 text-gray-800 border-gray-300';

    return (
      <div className="mb-6">
        <div className={`p-6 rounded-lg border-2 ${colorClass}`}>
          <h3 className="text-2xl font-bold mb-2">PEP Risk Assessment</h3>
          <div className="flex items-baseline gap-4 mb-4">
            <div className="text-5xl font-bold">{score?.toFixed(1)}%</div>
            <div className="text-xl font-semibold uppercase">{category} Risk</div>
          </div>
          
          {prediction.top_risk_factors && prediction.top_risk_factors.length > 0 && (
            <div className="mt-4">
              <h4 className="font-semibold mb-2">Top Contributing Factors:</h4>
              <ul className="list-disc list-inside">
                {prediction.top_risk_factors.map(([factor, contribution], idx) => (
                  <li key={idx} className="text-sm">
                    {factor.replace(/_/g, ' ')}: {contribution > 0 ? '+' : ''}{contribution.toFixed(1)} points
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
    <>
      {showResults && (
        <div>
          {procedureType === 'col' ? (
            <>
              <div className="card bg-base-100 shadow-xl mb-4">
                <div className="card-body">
                  <h2 className="card-title">Colonoscopy Results</h2>
                  <div className="overflow-x-auto">
                    <DataTable data={colonoscopyData} />
                  </div>
                </div>
              </div>

              <div className="card bg-base-100 shadow-xl mb-4">
                <div className="card-body">
                  <h2 className="card-title">Polyps Detected</h2>
                  <div className="overflow-x-auto">
                    <PolypsTable polyps={polypsData} />
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="card bg-base-100 shadow-xl mb-4">
              <div className="card-body">
                <h2 className="card-title">Procedure Results</h2>
                <div className="overflow-x-auto">
                  <DataTable data={procedureData} />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {showPEPRiskResults && (
        <>
          <div className="card bg-base-100 shadow mb-4">
            <div className="card-body">
              {renderPEPRiskPrediction()}

              {pepRiskData?.manual_input && Object.keys(pepRiskData.manual_input).length > 0 && (
                <div className="mb-6">
                  <h3 className="text-xl font-bold mb-3">Manually Input Risk Factors</h3>
                  <div className="overflow-x-auto">
                    <DataTable data={pepRiskData.manual_input} />
                  </div>
                </div>
              )}

              {pepRiskData?.llm_extracted && (
                <div className="mb-6">
                  <h3 className="text-xl font-bold mb-3">LLM-Extracted Risk Factors</h3>
                  <div className="overflow-x-auto">
                    <DataTable data={pepRiskData.llm_extracted} />
                  </div>
                </div>
              )}

              {pepRiskData?.prediction?.combined_risk_factors && (
                <div>
                  <h3 className="text-xl font-bold mb-3">All Combined Risk Factors</h3>
                  <div className="overflow-x-auto">
                    <DataTable data={pepRiskData.prediction.combined_risk_factors} />
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </>
  );
}
