#!/bin/bash
set -e

echo "Starting local deployment..."
docker-compose -f deployment/docker-compose.yml up --build -d
echo "Deployment successful. API running at http://localhost:8000"
