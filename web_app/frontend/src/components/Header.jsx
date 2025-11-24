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
    <div className="flex justify-between items-center mb-6">
      <h1 className="text-3xl font-bold">endoscribe</h1>
      <span className={`badge ${statusClass}`}>{statusText}</span>
    </div>
  );
}
