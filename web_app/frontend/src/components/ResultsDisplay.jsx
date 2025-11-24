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
  showResults,
  showPEPRiskResults
}) {
  if (!showResults && !showPEPRiskResults) {
    return null;
  }

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
        <div className="card bg-base-100 shadow-xl mb-4">
          <div className="card-body">
            <h2 className="card-title">PEP Risk Factors Extracted</h2>
            <div className="overflow-x-auto">
              <DataTable data={pepRiskData} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
