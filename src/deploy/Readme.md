# Financial Sentiment Analysis LSTM Model - Deployment

## Overview

This folder contains all the files used in the deployment process of this project.

## Deployment Structure

### Subdirectory Structure:
- **containerization/**
  - Contains the Dockerfile and container configuration
  - Build environment definitions and container optimization settings
  - Note: the .dockerignore file lives in the project root and the build command should be executed from there
- **frontend/**
  - React application
  - Components for text input, result visualization
  - Firebase hosting config
- **inference_pipeline/**
  - Inference optimization code and model serving utilities
  - Model and serialized pipelines in inference_pipeline/artifacts (not added to version control due to high size)
- **server.py**
  - FastAPI server implementation for the backend
  - API endpoint definitions and request handlers
  - CORS configuration and rate limiting logic


## Deployment Stack

### Cloud Provider
- **GCP (Google Cloud Platform)**
  - Region: `us-central1`

### Container Storage & Serving
- **Cost-effective Option**
  - GCP Artifact Registry
  - GCP Cloud Run
    
- **Production/Enterprise Option**
  - GCP Vertex AI. why ?
    - Model Registry for versioning and lineage tracking
    - Managed endpoints with auto-scaling
    - Advanced monitoring and explainability features

### Model Serving
- **FastAPI**

### Containerization
- **Docker**
  - Base image: tensorflow cpu version (lightest)

### Frontend
- **Firebase Hosting**

### Logging & Monitoring
- **GCP Cloud Run Logs Explorer**
  - Integration with Cloud Monitoring

## Scaling Considerations
- Configured to scale to zero when idle to minimize costs
- Maximum concurrent requests per instance: 20
- Memory allocation: 2GB per instance
- CPU allocation: 2 vCPU per instance

## Troubleshooting

### Common Issues
1. **Cold Start Latency**
   - Solution: Use minimum instance setting of 1 for critical applications
   
2. **CORS Issues with Frontend**
   - Solution: Update the allowed origins in server.py configuration


## Architecture Diagram
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   FastAPI   │────▶│ LSTM Model  │
│  React App  │◀────│   Server    │◀────│  Inference  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐ 
│   Firebase  │     │  Cloud Run  │
│   Hosting   │     │  Instance   │ 
└─────────────┘     └─────────────┘     
                           │
                           ▼
                    ┌──────────────┐
                    │ Cloud Logging│
                    │ & Monitoring │
                    └──────────────┘
```


## Alternative Deployment Options
- AWS: ECS/Fargate + Lambda + S3 + CloudFront
- Azure: Container Instances + Blob Storage + App Service
- Self-hosted: Kubernetes + Nginx + MinIO