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
      <div className="items-center mb-2">
        <h1 className="text-3xl font-bold">EndoScribe</h1>
      </div>
      <h3 className="mb-4"> AI-powered note writing</h3>
    </div>
    
  );
}
