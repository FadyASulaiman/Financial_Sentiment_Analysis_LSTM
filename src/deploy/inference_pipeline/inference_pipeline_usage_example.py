
import pickle
import os

from .prod_logger import logger

def create_production_inference_pipeline(
    fe_pipeline_path: str,
    preprocessing_pipeline_path: str,
    data_prep_pipeline_path: str,
    model_path: str,
    output_dir: str
):
    """
    Create a production-ready inference configuration that chains all pipelines
    and saves it as a pickle file for deployment
    """
    # Create a dictionary of paths to all pipeline components
    pipeline_paths = {
        'fe_pipeline_path': fe_pipeline_path,
        'preprocessing_pipeline_path': preprocessing_pipeline_path,
        'data_prep_pipeline_path': data_prep_pipeline_path,
        'model_path': model_path
    }
    
    # Save the pipeline paths as a pickle file
    pipeline_paths_file = os.path.join(output_dir, 'pipeline_paths.pkl')
    with open(pipeline_paths_file, 'wb') as f:
        pickle.dump(pipeline_paths, f)
    
    logger.info(f"Pipeline paths saved to {pipeline_paths_file}")
    
    # Create a wrapper function that loads the paths and calls the inference pipeline
    def create_inference_wrapper():
        """
        Create a wrapper function that loads the pipeline paths and provides
        a simple interface for single-string inference
        """

        wrapper_code = """
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
    \"\"\"
    Analyze the sentiment of a text string
    
    Args:
        text: String containing text to analyze
        
    Returns:
        Dictionary with sentiment analysis results
    \"\"\"
    # Load the pipeline paths
    with open(PIPELINE_PATHS_FILE, 'rb') as f:
        pipeline_paths = pickle.load(f)
    
    # Call the inference pipeline
    return production_inference_pipeline(text, pipeline_paths)
                """
        # Save the wrapper code
        wrapper_file = os.path.join(output_dir, 'sentiment_analyzer.py')
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        logger.info(f"Sentiment analyzer wrapper saved to {wrapper_file}")
    
    # Create the wrapper
    create_inference_wrapper()
    
    # Also save a simple example of how to use it with a single string
    example_code = """
# Example usage of the production inference pipeline with a single string
from sentiment_analyzer import analyze_sentiment

# Input a single text string
input_text = "Company XYZ reported better than expected earnings, with quarterly revenue up 15% year-over-year."

# Get prediction for the single string
result = analyze_sentiment(input_text)
print(f"Predicted sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
"""
    
    example_path = os.path.join(output_dir, 'inference_example.py')
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    logger.info(f"Example usage code saved to {example_path}")

if __name__ == "__main__":
        fe_pipeline_path = "data/feature_pipeline.pkl"
        preprocessing_pipeline_path = "data/cleaning_pipeline.pkl"
        data_prep_pipeline_path = "data/processed/data_prep_pipeline.pkl"
        model_path = "output/model/lstm_sentiment_80_acc.keras"
        output_dir = "output/prod_inference"
    
        create_production_inference_pipeline(
            fe_pipeline_path=fe_pipeline_path,
            preprocessing_pipeline_path=preprocessing_pipeline_path,
            data_prep_pipeline_path=data_prep_pipeline_path,
            model_path=model_path,
            output_dir=output_dir
        )