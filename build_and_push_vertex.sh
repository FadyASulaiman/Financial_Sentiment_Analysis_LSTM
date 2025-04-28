# GCP project variables
PROJECT_ID="sentiment-analysis-proj"  # Replace with your actual project ID
REGION="us-central1"          # Choose your preferred region
REPOSITORY="vertex-ai-models"
IMAGE_NAME="keras-lstm-sentiment-analysis-model"
TAG="v1"

# Dockerfile location
cd src/deploy/containerization

# Create repository in Artifact Registry (run only once)
gcloud artifacts repositories create $REPOSITORY \
  --repository-format=docker \
  --location=$REGION \
  --description="Docker repository for ML models"

# Build the Docker image
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$TAG .

# Authenticate Docker with GCP
gcloud auth configure-docker $REGION-docker.pkg.dev

# Push the image to Artifact Registry
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$TAG

# Register image with model registry


# Expose an endpoint to serve the model


# Deploy on cloudrun (If needed: uncomment & comment above two commands)

gcloud run deploy $SERVICE_NAME \
    --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated

gcloud run deploy sentiment-analysis-service \
  --image=us-central1-docker.pkg.dev/lstm-sentiment-analysis-457012/sentiment-analysis-lstm/first-build:v1.23 \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated