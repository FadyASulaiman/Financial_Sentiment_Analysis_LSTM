import os
from src.utils.feat_eng_pipeline_logger import logger
import pandas as pd

from src.pipelines.feature_engineering_pipleine import FeatureEngineeringPipeline

def main():
    """Main entry point for the feature engineering pipeline"""

    data_path = "data/data.csv"
    output_dir = os.path.dirname(data_path)
    config_path = "src/config/config.yaml"
    
    
    try:
        # Load data
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    
        # Drop NaN values
        data.dropna(subset=['Sentence'], inplace=True)

        # Create pipeline
        pipeline = FeatureEngineeringPipeline(config_path)
        
        # Fit and transform data
        transformed_data = pipeline.fit_transform(data)

        # Save transformed data  
        output_path = os.path.join(output_dir, 'feature_engineered_data.csv')
        transformed_data.to_csv(output_path, index=False)
        logger.info(f"Saved transformed data to {output_path}")
        
        # Save pipeline
        pipeline_path = os.path.join(
            output_dir,
            f"feature_pipeline_v{pipeline.version}.pkl"
        )
        pipeline.save_pipeline(pipeline_path)
        
        # Save config
        # config_path = os.path.join(
        #     os.path.dirname(output_dir),
        #     f"feature_config_v{pipeline.version}.yaml"
        # )
        # pipeline.save_config(config_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()