import os
from typing import List, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from src.deploy.inference_pipeline.sentiment_analyzer import analyze_sentiment

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - Modify in prod to allow requests from frontend domain only
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predictions: List[Any]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def prediction(request: TextRequest):
    """Prediction endpoint required by Vertex AI."""
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    predictions = analyze_sentiment(request.text)
    return PredictionResponse(predictions=[predictions])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))