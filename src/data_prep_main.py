import os
from src.utils.loggers.data_prep_pipeline_logger import logger


from src.pipelines.IV_data_prep_pipeline_orchestrator import DataPreparationOrchestrator


def prepare_data_for_lstm(data_path, output_dir=None):
    """Prepare data for LSTM model training"""
    try:
        # Configure output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(data_path), 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize
        orchestrator = DataPreparationOrchestrator(
            text_column='Sentence',
            label_column='Sentiment',
            max_sequence_length=128
        )
        
        # Process data
        logger.info(f"Preparing data from {data_path}")
        processed_df, X, y = orchestrator.preprocess(
            data_path=data_path,
            output_dir=output_dir
        )
        
        logger.info(f"Data preparation complete. Processed {len(processed_df)} samples.")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, orchestrator.mlflow_run_id
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise


if __name__ == "__main__":
    X, y, run_id = prepare_data_for_lstm(
        data_path="data/feature_engineered_data.csv",
        output_dir="data/processed"
    )
    
    print(f"Data preparation complete. MLflow run ID: {run_id}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")