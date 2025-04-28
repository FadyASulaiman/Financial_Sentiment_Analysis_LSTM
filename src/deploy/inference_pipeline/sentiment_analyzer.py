import pickle
import os

from .deployment_inference_pipeline_script import ProductionPredictor


def analyze_sentiment(text):

    # Call the inference pipeline
    p = ProductionPredictor()
    return p.production_inference_pipeline(text)
