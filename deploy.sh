#!/bin/bash
# Deployment script for Ticket Triage ML

set -e

echo "🚀 Deploying Ticket Triage ML..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if artifacts exist
if [ ! -f "artifacts/model.onnx" ]; then
    echo -e "${RED}❌ Error: Model artifacts not found!${NC}"
    echo "Please train the model first:"
    echo "  make train-fast  # Quick training (7 min)"
    echo "  make export      # Export to ONNX"
    exit 1
fi

echo -e "${GREEN}✅ Model artifacts found${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Stop existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker compose down 2>/dev/null || true

# Build and start services
echo -e "${YELLOW}🏗️  Building Docker image...${NC}"
docker compose build

echo -e "${YELLOW}🚀 Starting services...${NC}"
docker compose up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 5

# Check health
echo -e "${YELLOW}🔍 Checking service health...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API is healthy!${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Attempt $RETRY_COUNT/$MAX_RETRIES..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ API failed to start${NC}"
    echo "Check logs with: docker compose logs api"
    exit 1
fi

# Test prediction
echo -e "${YELLOW}🧪 Testing prediction...${NC}"
RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been trying to dispute incorrect information on my credit report"}')

if echo "$RESPONSE" | grep -q "topic"; then
    echo -e "${GREEN}✅ Prediction test passed!${NC}"
    echo "Response: $RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    echo -e "${RED}❌ Prediction test failed${NC}"
    echo "Response: $RESPONSE"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Deployment successful!${NC}"
echo ""
echo "Services:"
echo "  📡 API Server:  http://localhost:8000"
echo "  📊 MLflow UI:   http://localhost:8080"
echo "  📖 API Docs:    http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "  docker compose logs -f api    # View API logs"
echo "  docker compose logs -f mlflow # View MLflow logs"
echo "  docker compose down           # Stop all services"
echo "  make docker-logs              # View all logs"
echo ""
