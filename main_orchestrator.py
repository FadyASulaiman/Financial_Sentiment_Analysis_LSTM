import os
from src.pipelines.II_preprocessing_orchestrator import FinanceTextPreprocessingOrchestrator
from src.utils.preprocessing_logger import logger
from src.utils.constants import  MAX_SEQUENCE_LENGTH


def main():
    """Main entry point for the preprocessing pipeline."""

    data_path = "data/data.csv"
    output_dir = os.path.dirname(data_path)
    
    try:
        # Create preprocessor
        preprocessor = FinanceTextPreprocessingOrchestrator(max_sequence_length=MAX_SEQUENCE_LENGTH)

        # Preprocess data
        processed_df = preprocessor.preprocess(data_path=data_path, output_dir=output_dir)

        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()