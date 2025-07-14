#!/bin/bash
# Azure deployment script for Federated Health Risk Prediction System
# Prerequisites: az CLI, Docker, Azure Container Registry (ACR) created

RESOURCE_GROUP="federated-health-rg"
ACR_NAME="federatedhealthacr"
LOCATION="eastus"

# 1. Build Docker images
az acr build --registry $ACR_NAME --image federated-health-server:latest .

# 2. Push images to ACR (if not using az acr build)
# docker build -t $ACR_NAME.azurecr.io/federated-health-server:latest .
# docker push $ACR_NAME.azurecr.io/federated-health-server:latest

# 3. Deploy using Azure Container Instances (ACI)
az container create \
  --resource-group $RESOURCE_GROUP \
  --name federated-health-server \
  --image $ACR_NAME.azurecr.io/federated-health-server:latest \
  --cpu 2 --memory 4 \
  --registry-login-server $ACR_NAME.azurecr.io \
  --restart-policy OnFailure \
  --dns-name-label federated-health-server-$RANDOM \
  --ports 8080 8501

echo "Deployed to Azure Container Instances." 