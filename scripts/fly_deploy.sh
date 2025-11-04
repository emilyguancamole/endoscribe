#!/bin/bash
# Quick deployment script for EndoScribe on Fly.io
# Usage: ./scripts/fly_deploy.sh [app-name]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

APP_NAME="${1:-endoscribe}"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}EndoScribe Fly.io Deployment${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if flyctl is installed
if ! command -v fly &> /dev/null; then
    echo -e "${RED}Error: flyctl is not installed${NC}"
    echo "Install it from: https://fly.io/docs/hands-on/install-flyctl/"
    exit 1
fi

# Check if logged in
if ! fly auth whoami &> /dev/null; then
    echo -e "${YELLOW}Not logged in to Fly.io${NC}"
    echo "Logging in..."
    fly auth login
fi

echo -e "App name: ${GREEN}${APP_NAME}${NC}"
echo ""

# Check if app exists
if ! fly apps list | grep -q "^${APP_NAME}"; then
    echo -e "${YELLOW}App '${APP_NAME}' does not exist${NC}"
    echo "Creating app..."
    fly apps create "${APP_NAME}"
    echo ""
fi

# Check if volume exists
VOLUME_EXISTS=$(fly volumes list --app "${APP_NAME}" 2>/dev/null | grep -c "data" || true)
if [ "$VOLUME_EXISTS" -eq 0 ]; then
    echo -e "${YELLOW}Volume 'data' does not exist${NC}"
    echo "Creating 30GB volume with A10 GPU constraints..."
    echo ""
    echo -e "${YELLOW}NOTE: Fly.io will warn about using a single volume.${NC}"
    echo "Answer 'y' to proceed - a single volume is appropriate for this GPU deployment."
    echo ""
    fly volumes create data \
      --size 30 \
      --vm-gpu-kind a10 \
      --region ord \
      --app "${APP_NAME}"
    echo ""
fi

# Check if secrets are set
echo "Checking secrets..."
SECRETS_SET=$(fly secrets list --app "${APP_NAME}" 2>/dev/null | grep -c "ANTHROPIC_API_KEY" || true)
if [ "$SECRETS_SET" -eq 0 ]; then
    echo -e "${YELLOW}API keys not set${NC}"
    echo "Please set your API keys:"
    echo ""
    read -p "Enter ANTHROPIC_API_KEY: " ANTHROPIC_KEY

    if [ -n "$ANTHROPIC_KEY" ]; then
        fly secrets set ANTHROPIC_API_KEY="$ANTHROPIC_KEY" --app "${APP_NAME}"
    fi
    echo ""
fi

# Deploy
echo -e "${GREEN}Deploying to Fly.io...${NC}"
echo "This may take 10-15 minutes on first deployment."
echo ""

fly deploy --app "${APP_NAME}"

# Check deployment status
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "App URL: https://${APP_NAME}.fly.dev"
echo ""
echo "Useful commands:"
echo "  fly logs --app ${APP_NAME}           # View logs"
echo "  fly ssh console --app ${APP_NAME}    # SSH into container"
echo "  fly status --app ${APP_NAME}         # Check status"
echo "  fly apps open --app ${APP_NAME}      # Open in browser"
echo ""
