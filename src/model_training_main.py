import numpy as np
import tensorflow as tf

from src.pipelines.VI_model_training_pipeline_orchestrator import SentimentModelTrainer
from src.utils.model_eval.lstm_model_evaluators import SentimentModelEvaluator

from src.utils.loggers.model_training_and_eval_logger import logger


def main():
    """Main execution function"""

    embedding_dim = 100
    lstm_units = 128
    bidirectional = False
    batch_size = 32
    epochs = 14
    learning_rate = 0.01

    
    logger.info("Starting sentiment model training and evaluation")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Model config
    config = {
        'embedding_dim': embedding_dim,
        'lstm_units': lstm_units,
        'bidirectional': bidirectional,
        'dropout_rate': 0.3,
        'recurrent_dropout': 0.2,
        'batch_size': batch_size,
        'epochs': epochs,
        'patience': 5,
        'learning_rate': learning_rate,
        'validation_split': 0.1,
        'test_split': 0.1,
    }
    
    try:
        data_path = "data/processed/" + "processed_arrays.npz"
        model_dir = "output/model"

        trainer = SentimentModelTrainer(
            data_path= data_path,
            model_dir= model_dir
        )
        
        # Train model
        model, metrics, run_id = trainer.train(config)
        
        # Evaluate model
        logger.info("Loading test data for evaluation")
        X, y = trainer.load_data()
        
        # Use the same splits as in training
        from sklearn.model_selection import train_test_split
        
        # First, split off the validation+test data
        _, X_temp, _, y_temp = train_test_split(
            X, y, 
            test_size=config['validation_split'] + config['test_split'],
            stratify=y,
            random_state=42
        )
        
        # Then, split validation and test data
        test_size_adjusted = config['test_split'] / (config['validation_split'] + config['test_split'])
        _, X_test, _, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size_adjusted,
            stratify=y_temp,
            random_state=42
        )
        
        # Initialize evaluator
        eval_dir = "output/model_eval"

        evaluator = SentimentModelEvaluator(
            model=model,
            output_dir= eval_dir
        )
        
        report = evaluator.evaluate(X_test, y_test, run_id=run_id)
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()