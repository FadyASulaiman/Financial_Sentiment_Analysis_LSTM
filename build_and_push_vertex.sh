# Set your GCP project variables
PROJECT_ID="your-project-id"  # Replace with your actual project ID
REGION="us-central1"          # Choose your preferred region
REPOSITORY="vertex-ai-models"
IMAGE_NAME="keras-model-with-pipelines"
TAG="v1"

# Navigate to your project directory
cd path/to/deploy-project

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