#!/bin/bash
# AWS deployment script for Federated Health Risk Prediction System
# Prerequisites: AWS CLI, Docker, ECR repository created

AWS_REGION="us-east-1"
ECR_REPO="federated-health-server"

# 1. Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# 2. Build Docker image
docker build -t $ECR_REPO:latest .

# 3. Tag and push image
docker tag $ECR_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# 4. Deploy to ECS (Fargate)
# (Assumes ECS cluster and task definition are already set up)
aws ecs update-service --cluster federated-health-cluster --service federated-health-service --force-new-deployment

echo "Deployed to AWS ECS Fargate." 