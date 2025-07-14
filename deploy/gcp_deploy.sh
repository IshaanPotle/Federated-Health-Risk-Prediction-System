#!/bin/bash
# GCP deployment script for Federated Health Risk Prediction System
# Prerequisites: gcloud CLI, Docker, Google Artifact Registry enabled

PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
REPO="federated-health-repo"

# 1. Build Docker image
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/federated-health-server:latest .

# 2. Deploy to Cloud Run
gcloud run deploy federated-health-server \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/federated-health-server:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080

echo "Deployed to Google Cloud Run." 