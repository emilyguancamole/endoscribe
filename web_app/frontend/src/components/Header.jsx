export function Header({ healthStatus }) {
  const statusClass = {
    healthy: 'badge-success',
    degraded: 'badge-warning',
    error: 'badge-error'
  }[healthStatus] || 'badge-neutral';

  const statusText = {
    healthy: 'System Ready',
    degraded: 'System Degraded',
    error: 'System Error'
  }[healthStatus] || 'Checking...';

  return (
    <div>
      <div className={`fixed right-4 badge ${statusClass}`}>{statusText}</div>
      <div className="mt-4 mb-4">
        <h1 className="justify-center text-3xl font-bold">EndoScribe 𓂃✍︎</h1>
        <h3 className="justify-center mb-4">AI-powered clinical scribe for endoscopy</h3>
      </div>
      
    </div>
    
  );
}
