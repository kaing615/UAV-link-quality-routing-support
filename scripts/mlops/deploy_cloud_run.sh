#!/bin/bash
set -euo pipefail

PROJECT_ID=${1:-uav-gnn-support-routing}
REGION=${2:-us-central1}
API_SERVICE_NAME="uav-gnn-api"
DASHBOARD_SERVICE_NAME="uav-gnn-dashboard"
REPO_NAME="uav-gnn-repo"

echo "Deploying to Project: $PROJECT_ID in Region: $REGION"

echo "Enabling Cloud Run, Artifact Registry, and Cloud Build APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project "$PROJECT_ID"

if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" --project="$PROJECT_ID" > /dev/null 2>&1; then
    echo "Creating Artifact Registry repository $REPO_NAME..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Docker repository for UAV-GNN" \
        --project="$PROJECT_ID"
fi

if [[ ! -f models/best_model.pt ]]; then
    echo "Staging model weights..."
    bash scripts/mlops/stage_serving_model.sh
fi

API_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${API_SERVICE_NAME}:latest"
DASHBOARD_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${DASHBOARD_SERVICE_NAME}:latest"

deploy_with_retry() {
    local service_name="$1"
    shift

    for attempt in 1 2 3; do
        if gcloud run deploy "$service_name" "$@"; then
            return 0
        fi

        if [[ "$attempt" -lt 3 ]]; then
            echo "Cloud Run deploy failed, retrying in 20s..."
            sleep 20
        fi
    done

    echo "Cloud Run deploy failed after 3 attempts." >&2
    return 1
}

echo "Building API Docker image..."
cat > cloudbuild-api.yaml <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["pull", "${API_IMAGE}"]
    allowFailure: true

  - name: gcr.io/cloud-builders/docker
    args: ["build", "--cache-from", "${API_IMAGE}", "-f", "Dockerfile.serve", "-t", "${API_IMAGE}", "."]
images:
  - "${API_IMAGE}"
EOF

gcloud builds submit . --config cloudbuild-api.yaml --project "$PROJECT_ID"

echo "Deploying API to Cloud Run..."
deploy_with_retry "$API_SERVICE_NAME" \
    --image "$API_IMAGE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1

API_URL=$(gcloud run services describe "$API_SERVICE_NAME" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format "value(status.url)")

echo "API deployed at: $API_URL"

echo "Building Dashboard Docker image..."
cat > cloudbuild-dashboard.yaml <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["pull", "${DASHBOARD_IMAGE}"]
    allowFailure: true

  - name: gcr.io/cloud-builders/docker
    args: ["build", "--cache-from", "${DASHBOARD_IMAGE}", "-f", "Dockerfile.dashboard", "-t", "${DASHBOARD_IMAGE}", "."]
images:
  - "${DASHBOARD_IMAGE}"
EOF

gcloud builds submit . --config cloudbuild-dashboard.yaml --project "$PROJECT_ID"

echo "Deploying Dashboard to Cloud Run..."
deploy_with_retry "$DASHBOARD_SERVICE_NAME" \
    --image "$DASHBOARD_IMAGE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --allow-unauthenticated \
    --set-env-vars="API_URL=$API_URL"

DASHBOARD_URL=$(gcloud run services describe "$DASHBOARD_SERVICE_NAME" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format "value(status.url)")

echo "Dashboard deployed at: $DASHBOARD_URL"
echo "Deployment successful!"
