import pandas as pd
from src.pipelines.I_eda_pipeline_orchestrator import FinanceSentimentEDAOrchestrator
from src.utils.loggers.eda_logger import logger

def main():
    """Main entry point for the application"""
    try:
        # Use static file path and output directory
        file_path = "data/data.csv"
        output_dir = "Exploratory Analysis Output"
        
        # Initialize and run EDA with static file path
        eda = FinanceSentimentEDAOrchestrator(file_path=file_path, output_dir=output_dir)

        # Run the EDA pipeline
        processed_data = eda.run_eda()
        
        logger.info(f"EDA completed successfully. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()