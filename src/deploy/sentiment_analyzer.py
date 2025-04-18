import pickle
import os
import sys

# Add the current directory to the path so we can import the inference function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the inference function from this module
from src.deploy.deployment_inference_pipeline_script import production_inference_pipeline

# Path to the pipeline paths file
PIPELINE_PATHS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_paths.pkl')

def analyze_sentiment(text):
    """
    Analyze the sentiment of a text string
    
    Args:
        text: String containing text to analyze
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Load the pipeline paths
    with open(PIPELINE_PATHS_FILE, 'rb') as f:
        pipeline_paths = pickle.load(f)
    
    # Call the inference pipeline
    return production_inference_pipeline(text, pipeline_paths)
    