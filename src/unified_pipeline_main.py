
import os

import pandas as pd
from src.deploy.inference_pipeline.prod_logger import logger
from src.pipelines.VII_prod_unified_pipeline import UnifiedPipeline



def unified_pipeline_main(data_path: str, 
                         config_path: str = None, 
                         output_pipeline_path: str = None, 
                         max_sequence_length: int = 128):
    """
    Main function to process data through the unified pipeline and optionally save the pipeline.
    
    Args:
        data_path: Path to input data CSV
        config_path: Path to configuration file
        output_pipeline_path: Path where the pipeline should be saved (optional)
        max_sequence_length: Maximum sequence length for text
        
    Returns:
        Tuple of (X, y) arrays ready for model training/inference
    """
    try:
        # Load data
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Drop NaN values in text column
        data.dropna(subset=['Sentence'], inplace=True)
        logger.info(f"After dropping NaN values: {data.shape[0]} rows")
        
        # Initialize unified pipeline
        pipeline = UnifiedPipeline(
            config_path=config_path,
            max_sequence_length=max_sequence_length
        )
        
        # Process data
        X, y = pipeline.process(data)
        
        # Save pipeline if requested
        if output_pipeline_path:
            pipeline.save(output_pipeline_path)
        
        return X, y, pipeline
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":

    data_path = "data/data.csv"
    config_path = "src/config/config.yaml"
    output_pipeline_path = "models/unified_pipeline.joblib"
    
    X, y, pipeline = unified_pipeline_main(
        data_path=data_path,
        config_path=config_path,
        output_pipeline_path=output_pipeline_path,
        max_sequence_length=128
    )
    
    print(f"Processing complete. Output shapes: X={X.shape}, y={y.shape}")