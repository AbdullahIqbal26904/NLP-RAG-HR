#!/bin/bash
# Quick deploy/redeploy script
# Run from the project root: bash deploy/deploy.sh

set -e

echo "Building Docker image..."
docker compose -f docker-compose.prod.yml build

echo "Stopping existing containers..."
docker compose -f docker-compose.prod.yml down 2>/dev/null || true

echo "Starting application..."
docker compose -f docker-compose.prod.yml up -d

echo "Waiting for app to start..."
sleep 10

# Health check
if curl -s -f http://localhost/_stcore/health > /dev/null 2>&1; then
    echo ""
    echo "App is running! Access it at:"
    echo "  Local:  http://localhost"
    # Try to get public IP
    PUBLIC_IP=$(curl -s --max-time 3 http://checkip.amazonaws.com 2>/dev/null || echo "")
    if [ -n "$PUBLIC_IP" ]; then
        echo "  Public: http://$PUBLIC_IP"
    fi
else
    echo ""
    echo "App may still be loading (models downloading). Check logs:"
    echo "  docker compose -f docker-compose.prod.yml logs -f"
fi
