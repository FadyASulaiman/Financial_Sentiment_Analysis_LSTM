import pickle
import os

from src.deploy.deployment_inference_pipeline_script import ProductionPredictor

PIPELINE_PATHS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_paths.pkl')

def analyze_sentiment(text):

    # Load pipeline paths
    with open(PIPELINE_PATHS_FILE, 'rb') as f:
        pipeline_paths = pickle.load(f)
    
    # Call the inference pipeline
    p = ProductionPredictor()
    return p.production_inference_pipeline(text, pipeline_paths)
    