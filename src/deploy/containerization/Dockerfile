FROM tensorflow/tensorflow:latest

# Install system dependencies for pandas and sklearn
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir --retries 2 -r requirements.txt
# install language models required for spacy
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m spacy download en_core_web_lg

# Copy model and pipelines
COPY src/deploy/containerization/artifacts/lstm_sentiment_80_acc.keras /app/lstm_sentiment_80_acc
COPY src/deploy/containerization/artifacts/feature_pipeline.pkl /app/feature_pipeline.pkl
COPY src/deploy/containerization/artifacts/cleaning_pipeline.pkl /app/cleaning_pipeline.pkl
COPY src/deploy/containerization/artifacts/data_prep_pipeline.pkl /app/data_prep_pipeline.pkl

# Copy application code
COPY src/deploy/inference_pipeline/ /app/inference_pipeline/
COPY src/deploy/server.py /app/server.py

WORKDIR /app

# Expose port 8080 for Vertex AI prediction endpoint
EXPOSE 8080

# Run FastAPI server with uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]