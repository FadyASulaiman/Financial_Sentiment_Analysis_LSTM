import os
from src.utils.constants import MAX_SEQUENCE_LENGTH
from src.utils.loggers.preprocessing_logger import logger
from src.pipelines.IV_minority_oversampling_pipeline import DataBalancingOrchestrator


def main():
    """Main entry point for the Data Balancing (Minority Oversampling) pipeline."""

    data_path = "data/cleaned_FE_data.csv"
    output_dir = os.path.dirname(data_path)
    
    try:
        # Create preprocessor
        preprocessor = DataBalancingOrchestrator()

        # Preprocess data
        processed_df = preprocessor.preprocess(data_path=data_path, output_dir=output_dir)

        logger.info("Data Balancing (Minority Oversampling) completed successfully")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()