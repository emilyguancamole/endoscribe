// API utilities
export async function processTranscript(transcript, procedureType, sessionId) {
  const response = await fetch('/api/process', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      transcript,
      procedure_type: procedureType,
      session_id: sessionId
    })
  });
  
  return response.json();
}

export async function checkHealth() {
  const response = await fetch('/health');
  return response.json();
}
