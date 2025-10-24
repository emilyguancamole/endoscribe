# Deploying EndoScribe to Fly.io with GPU Support

This guide walks through deploying the EndoScribe web application to Fly.io with NVIDIA A10 GPU support for WhisperX transcription.

## Prerequisites

1. **Fly.io Account**: Sign up at https://fly.io
2. **flyctl CLI**: Install the Fly.io command-line tool
   ```bash
   # macOS
   brew install flyctl

   # Linux
   curl -L https://fly.io/install.sh | sh

   # Windows
   iwr https://fly.io/install.ps1 -useb | iex
   ```
3. **Authentication**: Log in to Fly.io
   ```bash
   fly auth login
   ```

## Architecture Overview

The deployment uses:
- **NVIDIA A10 GPU** (24GB VRAM) - Sufficient for WhisperX large-v3 model
- **Persistent Volume** (30GB) - Stores models, uploads, and results
- **Auto-scale to Zero** - Reduces costs when idle
- **Ubuntu 22.04 + CUDA 12.2** - Base container with GPU support

## Step-by-Step Deployment

### 1. Prepare Environment Variables

Create a `.env` file locally (not committed to git) with your API keys:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional (if using other services)
HF_TOKEN=hf_...
OPENAI_API_KEY=sk-...
```

### 2. Configure Your App Name

Edit `fly.toml` and set your unique app name:

```toml
app = "your-app-name-here"  # Change this to your desired app name
```

### 3. Create the Fly.io App

```bash
# Create the app without deploying yet
fly apps create your-app-name-here --org personal
```

### 4. Create Persistent Volume

Create a GPU-constrained volume for model caching and data storage:

```bash
fly volumes create data \
  --size 30 \
  --vm-gpu-kind a10 \
  --region ord \
  --app your-app-name-here
```

**Important Notes:**
- Volume size: 30GB (enough for WhisperX models ~5GB + data)
- Must use `--vm-gpu-kind a10` to match your VM type
- Region `ord` (Chicago) is currently the only region with A10 GPUs
- Cost: ~$0.15/GB/month (~$4.50/month for 30GB)

### 5. Set Secrets

Set your API keys as Fly.io secrets (encrypted environment variables):

```bash
fly secrets set \
  ANTHROPIC_API_KEY="your-api-key-here" \
  HF_TOKEN="your-hf-token-here" \
  --app your-app-name-here
```

### 6. Deploy the Application

```bash
# Deploy from the project root
fly deploy --app your-app-name-here
```

**First deployment notes:**
- Build will take 10-15 minutes (installing CUDA, Python packages, etc.)
- WhisperX models (~5GB) will download on first startup
- Initial startup will take 5-10 minutes for model downloads
- Subsequent starts will be faster (models cached in volume)

### 7. Monitor Deployment

Watch the deployment logs:

```bash
fly logs --app your-app-name-here
```

Look for these success messages:
```
WhisperX model loaded successfully!
WhisperX alignment model loaded successfully!
LLM handler initialized successfully
All processors initialized successfully
```

### 8. Access Your Application

```bash
# Open in browser
fly apps open --app your-app-name-here

# Or get the URL
fly info --app your-app-name-here
```

Your app will be available at: `https://your-app-name-here.fly.dev`

## Verify Deployment

### Check Health Endpoint

```bash
curl https://your-app-name-here.fly.dev/health
```

Expected response:
```json
{
  "status": "healthy",
  "whisper_loaded": true,
  "llm_initialized": true,
  "supported_procedures": ["col", "eus", "ercp", "egd"]
}
```

### Test Transcription

1. Open the web interface
2. Select a procedure type (e.g., "Colonoscopy")
3. Click "Start Recording"
4. Grant microphone permissions
5. Speak a test phrase
6. Verify transcription appears in real-time
7. Click "Stop" and "Submit for Processing"
8. Verify structured data extraction works

## Auto-Scaling Behavior

The app is configured to **scale to zero** when idle:

- **Active state**: When receiving requests, one A10 GPU machine runs (~$2.08/hour)
- **Idle state**: After ~5 minutes of no traffic, machine automatically stops ($0/hour)
- **Wake-up**: Incoming requests automatically start the machine
- **Cold start time**: ~30-60 seconds (models loaded from volume, not downloaded)

### Cost Implications

- **Variable usage**: Only pay when transcribing (~$2.08/hour active)
- **Persistent storage**: ~$4.50/month for 30GB volume (always billed)
- **Network egress**: Usually negligible for this application

**Example monthly costs:**
- Light usage (10 hours/month): ~$25/month
- Medium usage (50 hours/month): ~$110/month
- Heavy usage (200 hours/month): ~$420/month
- 24/7 operation: ~$1,500/month

## Management Commands

### View Logs

```bash
# Real-time logs
fly logs --app your-app-name-here

# Filter for errors
fly logs --app your-app-name-here | grep ERROR
```

### SSH into Container

```bash
fly ssh console --app your-app-name-here

# Check GPU
nvidia-smi

# Check volumes
df -h /data

# Check processes
ps aux | grep python
```

### Scale Manually

```bash
# Force machine to stay running (disable auto-stop)
fly scale count 1 --app your-app-name-here

# Force machine to stop
fly scale count 0 --app your-app-name-here
```

### Update Secrets

```bash
fly secrets set ANTHROPIC_API_KEY="new-key" --app your-app-name-here
```

### Redeploy After Changes

```bash
# After modifying code
fly deploy --app your-app-name-here

# Force rebuild without cache
fly deploy --no-cache --app your-app-name-here
```

## Volume Management

### Check Volume Status

```bash
fly volumes list --app your-app-name-here
```

### Access Volume Data

```bash
# SSH into machine
fly ssh console --app your-app-name-here

# Navigate to volume
cd /data

# Check space usage
du -sh *
```

### Backup Volume

```bash
# Create snapshot
fly volumes snapshots create <volume-id> --app your-app-name-here

# List snapshots
fly volumes snapshots list <volume-id> --app your-app-name-here
```

### Expand Volume (if needed)

```bash
fly volumes extend <volume-id> --size 50 --app your-app-name-here
```

## Troubleshooting

### Issue: Build Fails with CUDA Errors

**Solution**: Ensure Dockerfile uses correct CUDA base image
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
```

### Issue: WhisperX Model Download Fails

**Symptoms**: Logs show "Failed to load WhisperX model"

**Solutions**:
1. Check HF_TOKEN secret is set
2. Verify volume has sufficient space (10GB+ free)
3. Check internet connectivity from container

### Issue: Machine Won't Start

**Symptoms**: Health checks failing, machine restarts

**Solutions**:
```bash
# Check logs for errors
fly logs --app your-app-name-here

# SSH to debug
fly ssh console --app your-app-name-here

# Check volume mounted correctly
ls -la /data
```

### Issue: Transcription is Slow

**Symptoms**: Real-time transcription lags

**Solutions**:
1. Verify GPU is being used:
   ```bash
   fly ssh console
   nvidia-smi
   ```
2. Check if model is loaded from volume (fast) or downloading (slow)
3. Consider upgrading to L40S GPU if A10 is insufficient

### Issue: High Costs

**Symptoms**: Unexpected bills

**Solutions**:
1. Verify auto-stop is working:
   ```bash
   fly status --app your-app-name-here
   ```
2. Check machine stops when idle (5 minutes of no traffic)
3. Review volume size (can reduce if not needed)

### Issue: WebSocket Disconnects

**Symptoms**: Transcription stops mid-recording

**Solutions**:
1. Fly.io proxy handles WebSockets automatically
2. Check client-side errors in browser console
3. Verify audio chunks aren't too large (3-second chunks recommended)

## Advanced Configuration

### Use L40S GPU (Better Performance)

Edit `fly.toml`:
```toml
[vm]
  size = "l40s"  # 48GB VRAM, ~$3/hour
```

Recreate volume:
```bash
fly volumes create data --size 30 --vm-gpu-kind l40s --region ord
```

### Disable Auto-Scaling (Keep Always Running)

Edit `fly.toml`:
```toml
[http_service]
  auto_stop_machines = false
  auto_start_machines = false
  min_machines_running = 1
```

### Add Custom Domain

```bash
# Add SSL certificate
fly certs add endoscribe.yourdomain.com --app your-app-name-here

# Configure DNS
# Add CNAME record: endoscribe.yourdomain.com -> your-app-name-here.fly.dev
```

### Enable HTTPS-Only

Already enabled by default in `fly.toml`:
```toml
[http_service]
  force_https = true
```

### Add Monitoring

```bash
# View metrics
fly dashboard --app your-app-name-here

# Setup alerts (via Fly.io dashboard)
```

## Security Considerations

1. **API Keys**: Always use `fly secrets`, never commit to git
2. **HTTPS**: Enforced by default for all connections
3. **WebSocket Security**: Fly.io proxy handles SSL termination
4. **Volume Encryption**: Volumes are encrypted at rest
5. **Network Isolation**: Machines run in isolated network

## Cleanup / Teardown

To completely remove the deployment:

```bash
# Destroy app (also removes volumes and data)
fly apps destroy your-app-name-here

# Confirm deletion
# Type the app name when prompted
```

**Warning**: This permanently deletes all data in volumes!

## Support & Resources

- **Fly.io Docs**: https://fly.io/docs/
- **Fly.io GPU Guide**: https://fly.io/docs/gpus/gpu-quickstart/
- **Fly.io Community**: https://community.fly.io/
- **EndoScribe Issues**: https://github.com/emilyguancamole/endoscribe/issues

## Next Steps

After successful deployment:

1. **Test thoroughly**: Record sample procedures, verify accuracy
2. **Monitor costs**: Check Fly.io dashboard regularly
3. **Set up alerts**: Configure health check alerts
4. **Plan backups**: Schedule volume snapshots if needed
5. **Document workflows**: Create runbooks for your team

## Development vs Production

The application automatically detects Fly.io environment via `FLY_APP_NAME` env var:

- **Production** (Fly.io): Uses `/data/*` persistent volumes
- **Development** (Local): Uses `web_app/uploads`, `web_app/results`

This allows you to develop locally without any code changes.
