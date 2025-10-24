# Scale-to-Zero Implementation for EndoScribe

## Overview

EndoScribe implements **app-level idle shutdown** to minimize GPU costs on Fly.io. The application automatically exits after 60 seconds of inactivity, allowing the expensive A10 GPU machine to stop and avoid charges.

## Cost Savings

**Without scale-to-zero:**
- A10 GPU: $2.08/hour × 24 hours = $49.92/day = $1,497.60/month

**With scale-to-zero (assuming 50 hours/month usage):**
- Active time: 50 hours × $2.08 = $104.00
- Volume storage: ~$4.50/month
- **Total: ~$110/month**
- **Savings: ~$1,390/month (93% reduction!)**

**Compared to Fly.io's default 5-minute timeout:**
- Default wastes ~5 minutes per session
- Custom 60s timeout wastes only 1 minute
- For 200 sessions/month: saves ~13 hours = $27/month

## How It Works

### 1. Activity Tracking

Every user interaction updates a global timestamp:

```python
last_activity_time = time.time()  # Updated on every request

def update_activity():
    """Called on every WebSocket message, API call, and page load"""
    global last_activity_time
    last_activity_time = time.time()
```

**Tracked activities:**
- WebSocket connections and messages (transcription)
- `/api/process` calls (LLM extraction)
- Page loads (`/` homepage)

**NOT tracked (intentionally):**
- `/health` checks (would prevent shutdown)
- `/gpu-info` diagnostics
- Static file requests

### 2. Background Monitor

A background task checks idle time every 10 seconds:

```python
async def check_idle_and_shutdown():
    while True:
        await asyncio.sleep(10)
        idle_time = time.time() - last_activity_time

        if idle_time >= IDLE_TIMEOUT_SECONDS:
            print("Idle shutdown triggered!")
            os.kill(os.getpid(), signal.SIGTERM)
            break
```

### 3. Graceful Shutdown

When idle threshold is reached:
1. Process sends `SIGTERM` to itself
2. FastAPI's lifespan cleanup runs
3. Process exits with code 0 (success)
4. Fly.io machine stops automatically

### 4. Auto-Restart

Fly Proxy handles wake-up:
1. New HTTP request arrives at proxy
2. Proxy detects machine is stopped
3. Proxy starts the machine automatically
4. Request is forwarded once app is ready
5. User sees ~30-60 second delay (cold start)

## Configuration

### Environment Variables

**`IDLE_TIMEOUT_SECONDS`** (default: 60)
```bash
# In fly.toml or via fly secrets
IDLE_TIMEOUT_SECONDS=120  # 2 minutes
```

Shorter = more aggressive cost savings, more cold starts
Longer = fewer cold starts, higher idle costs

**Auto-detection:**
- Idle shutdown **enabled** on Fly.io (detected via `FLY_APP_NAME`)
- Idle shutdown **disabled** locally (development convenience)

### fly.toml Settings

```toml
[http_service]
  auto_stop_machines = "stop"   # Let app control shutdown
  auto_start_machines = true     # Fly Proxy restarts on traffic
  min_machines_running = 0       # Allow full scale-to-zero

  [http_service.concurrency]
    soft_limit = 1  # Wake single machine immediately
```

**Key changes from defaults:**
- `soft_limit = 1` (was 5): Faster wake-ups for single-user scenarios
- **Removed health checks**: Would keep machine alive unnecessarily

## Monitoring

### Check if Idle Shutdown is Active

```bash
# View logs to see idle monitor status
fly logs --app endo2

# Look for:
# "Idle shutdown enabled: will exit after 60s of inactivity"
# "Starting idle shutdown monitor (60s timeout)..."
```

### Watch for Shutdowns

```bash
# Monitor logs for idle shutdown events
fly logs --app endo2 | grep "IDLE SHUTDOWN"

# Expected output when shutting down:
# ============================================================
# IDLE SHUTDOWN TRIGGERED
# ============================================================
# Idle time: 61.2s (threshold: 60s)
# Shutting down to save GPU costs...
# Fly Proxy will restart on next request
# ============================================================
```

### Check Machine Status

```bash
# See if machine is running or stopped
fly status --app endo2

# Output shows:
# Machines
# PROCESS ID              VERSION REGION  STATE   CHECKS  LAST UPDATED
# app     91857143c02ee8  12      ord     stopped         2024-10-24T20:15:33Z
```

## Testing Locally

The idle shutdown is **automatically disabled** when running locally (no `FLY_APP_NAME` env var).

To test locally:
```bash
# Set FLY_APP_NAME to enable idle shutdown
export FLY_APP_NAME=test
export IDLE_TIMEOUT_SECONDS=30

# Run server
python web_app/server.py

# Expected output:
# "Idle shutdown enabled: will exit after 30s of inactivity"
# "Starting idle shutdown monitor (30s timeout)..."

# Wait 30 seconds without making requests
# Server will automatically exit
```

## Deployment

After making changes, redeploy:

```bash
fly deploy --app endo2
```

Watch logs to verify:
```bash
fly logs --app endo2

# Look for:
# "Idle shutdown enabled: will exit after 60s of inactivity"
# "Starting idle shutdown monitor (60s timeout)..."
```

## Troubleshooting

### Machine Not Shutting Down

**Symptom:** Machine stays running even with no activity

**Causes:**
1. Health checks still enabled (remove from `fly.toml`)
2. WebSocket connection left open (client should close)
3. Background jobs keeping process active

**Fix:**
```bash
# Check logs for activity
fly logs --app endo2 | grep "update_activity"

# Verify no health checks
grep "checks" fly.toml  # Should return nothing
```

### Machine Takes Too Long to Shut Down

**Symptom:** Idle time > 60s but no shutdown

**Cause:** Background monitor runs every 10s, so actual shutdown time is 60-70s

**This is normal:** The check interval prevents constant CPU usage

### Cold Starts Too Slow

**Symptom:** Users complain about 30-60s wait when machine is stopped

**Solutions:**
1. **Increase idle timeout** to reduce shutdown frequency:
   ```bash
   fly secrets set IDLE_TIMEOUT_SECONDS=300  # 5 minutes
   ```

2. **Keep machine running during peak hours:**
   ```bash
   fly scale count 1  # Temporarily disable scale-to-zero
   ```

3. **Pre-warm before expected usage:**
   ```bash
   curl https://endo2.fly.dev  # Wake up machine
   ```

## Best Practices

### For Development
- Disable idle shutdown locally (default behavior)
- Test with short timeout (30s) before deploying
- Monitor logs after deployment to verify behavior

### For Production
- **Low usage** (<10 sessions/day): 60s timeout, embrace cold starts
- **Medium usage** (10-50 sessions/day): 120-300s timeout for balance
- **High usage** (>50 sessions/day): Consider keeping 1 machine running 24/7

### Cost Optimization Tips

1. **Batch transcriptions:** Process multiple audio files in one session
2. **Pre-warn users:** Display "Waking up GPU..." message during cold start
3. **Monitor usage patterns:** Adjust timeout based on actual usage
4. **Weekend schedules:** Longer timeout during work hours, shorter on weekends

## Comparison with Alternatives

### Fly.io Default Auto-Stop (~5 minutes)
**Pros:**
- No code changes needed
- Handles complex scenarios automatically

**Cons:**
- Wastes ~$0.17 per session in idle time
- No control over timeout duration
- Health checks interfere with detection

### Our App-Level Shutdown (60 seconds)
**Pros:**
- Full control over idle timeout
- Faster shutdown = lower costs (~$40-50/month savings)
- Can customize per-endpoint if needed

**Cons:**
- Requires application code changes
- Must track activity carefully
- Need to test shutdown behavior

### Always Running (No Scale-to-Zero)
**Pros:**
- Zero cold starts
- Instant response times
- Simplest setup

**Cons:**
- $1,500/month even with no usage
- Wasteful for intermittent workloads

## Conclusion

App-level idle shutdown is the **optimal solution** for GPU-based inference workloads with sporadic usage. The 60-second timeout provides excellent cost savings while maintaining acceptable cold start times for typical endoscopy transcription use cases.

For 24/7 production deployments with constant traffic, consider disabling scale-to-zero and keeping machines running.
