// API utilities
export async function processTranscript(transcript, procedureType, sessionId, manualPepData = null) {
  const requestBody = {
    transcript,
    procedure_type: procedureType,
    session_id: sessionId
  };
  
  // Add manual PEP data if provided
  if (manualPepData) {
    requestBody.manual_pep_data = manualPepData;
  }
  
  const response = await fetch('/api/process', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(requestBody)
  });
  
  return response.json();
}

export async function checkHealth() {
  const response = await fetch('/health');
  return response.json();
}
