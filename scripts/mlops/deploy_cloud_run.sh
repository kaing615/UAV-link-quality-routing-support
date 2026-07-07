#!/bin/bash
set -e


PROJECT_ID=${1:-uav-gnn-support-routing}
REGION=${2:-us-central1}
API_SERVICE_NAME="uav-gnn-api"
DASHBOARD_SERVICE_NAME="uav-gnn-dashboard"
REPO_NAME="uav-gnn-repo"

echo "Deploying to Project: $PROJECT_ID in Region: $REGION"

echo "Enabling Cloud Run, Artifact Registry, and Cloud Build APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project $PROJECT_ID

if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION --project=$PROJECT_ID > /dev/null 2>&1; then
    echo "Creating Artifact Registry repository $REPO_NAME..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Docker repository for UAV-GNN" \
        --project=$PROJECT_ID
fi

echo "Staging model weights..."
bash scripts/mlops/stage_serving_model.sh edge-sage

API_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${API_SERVICE_NAME}:latest"
echo "Building API Docker image..."
gcloud builds submit --tag $API_IMAGE -f Dockerfile.serve . --project $PROJECT_ID

echo "Deploying API to Cloud Run..."
gcloud run deploy $API_SERVICE_NAME \
    --image $API_IMAGE \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1

API_URL=$(gcloud run services describe $API_SERVICE_NAME --platform managed --region $REGION --project $PROJECT_ID --format 'value(status.url)')
echo "API deployed at: $API_URL"

DASHBOARD_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${DASHBOARD_SERVICE_NAME}:latest"
echo "Building Dashboard Docker image..."
gcloud builds submit --tag $DASHBOARD_IMAGE -f Dockerfile.dashboard . --project $PROJECT_ID

echo "Deploying Dashboard to Cloud Run..."
gcloud run deploy $DASHBOARD_SERVICE_NAME \
    --image $DASHBOARD_IMAGE \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --set-env-vars="API_URL=$API_URL"

DASHBOARD_URL=$(gcloud run services describe $DASHBOARD_SERVICE_NAME --platform managed --region $REGION --project $PROJECT_ID --format 'value(status.url)')
echo "Dashboard deployed at: $DASHBOARD_URL"
echo "Deployment successful!"
