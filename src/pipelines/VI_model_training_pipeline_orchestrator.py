import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import tensorflow as tf

from src.models.lstm_model import LSTMSentimentModel

from src.utils.loggers.model_training_and_eval_logger import logger


class SentimentModelTrainer:
    """Class to handle model training workflow"""
    
    def __init__(self, data_path, model_dir='models'):
        """Initialize model trainer"""
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('sentiment_model')
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    def load_data(self):
        """Load preprocessed data"""
        logger.info(f"Loading data from {self.data_path}")
        
        data = np.load(self.data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        logger.info(f"Loaded data with shapes: X={X.shape}, y={y.shape}")
        
        # Log class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_distribution}")
        
        return X, y
            
        
    def prepare_embedding_matrix(self, word_index, embedding_path, embedding_dim=100):
        """Prepare embedding matrix from pre-trained embeddings
        
        Args:
            word_index: Word-to-index mapping
            embedding_path: Path to pre-trained embeddings
            embedding_dim: Embedding dimension
            
        Returns:
            Embedding matrix
        """
        logger.info(f"Loading embeddings from {embedding_path}")
        
        # Initialize embedding matrix
        vocab_size = len(word_index) + 1  # +1 for padding
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        # Load embeddings
        embeddings_index = {}
        with open(embedding_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        logger.info(f"Loaded {len(embeddings_index)} word vectors")
        
        # Create embedding matrix
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        # Calculate coverage
        found_words = sum(1 for word in word_index if word in embeddings_index)
        coverage = found_words / len(word_index) * 100
        logger.info(f"Embedding coverage: {coverage:.2f}% ({found_words}/{len(word_index)} words)")
        
        return embedding_matrix
    


    def train(self, config=None):
        """Train the model
        
        Args:
            config: Model configuration
        """
        if config is None:
            config = {
                'embedding_dim': 100,
                'lstm_units': 128,
                'bidirectional': True,
                'dropout_rate': 0.3,
                'recurrent_dropout': 0.2,
                'batch_size': 32,
                'epochs': 20,
                'patience': 5,
                'learning_rate': 0.001,
                'validation_split': 0.1,
                'test_split': 0.1
            }
        
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run with ID: {run_id}")
            
            # Log parameters
            for key, value in config.items():
                mlflow.log_param(key, value)
            
            # Load data
            X, y = self.load_data()
            
            # Split data into train, validation, and test sets
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, 
                test_size=config['validation_split'] + config['test_split'],
                stratify=y,
                random_state=42
            )
            
            # Further split temp data into validation and test
            test_size_adjusted = config['test_split'] / (config['validation_split'] + config['test_split'])
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=test_size_adjusted,
                stratify=y_temp,
                random_state=42
            )
            
            logger.info(f"Data split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
            
            # Compute class weights for imbalanced data
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            logger.info(f"Class weights: {class_weight_dict}")
            
            # Determine model parameters
            vocab_size = np.max(X) + 1  # Max index + 1
            max_sequence_length = X.shape[1]
            
            # Create model with 3 output classes
            model = LSTMSentimentModel(
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                embedding_dim=config['embedding_dim'],
                lstm_units=config['lstm_units'],
                bidirectional=config['bidirectional'],
                dropout_rate=config['dropout_rate'],
                recurrent_dropout=config['recurrent_dropout'],
                num_classes=3,  # 3 classes for positive, neutral, negative
                model_name=f"lstm_sentiment_{run_id[:8]}",
                log_dir=os.path.join(self.model_dir, 'logs')
            )
            
            # Compile model
            model.compile(
                learning_rate=config['learning_rate']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                X_val, y_val,
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                patience=config['patience'],
                class_weights=class_weight_dict
            )

            # Print model summary
            model.summary()
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Save model
            model_path = model.save(os.path.join(self.model_dir, f"lstm_sentiment_{run_id[:8]}.h5"))
            
            # Log to MLflow
            model.log_to_mlflow(run_id)
            
            logger.info(f"Model training completed. Model saved to {model_path}")
            return model, metrics, run_id
        
