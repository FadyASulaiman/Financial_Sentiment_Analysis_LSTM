from abc import ABC, abstractmethod
import os
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard # type: ignore
import mlflow.keras

from src.utils.loggers.model_training_and_eval_logger import logger

class ModelBase(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name, log_dir='logs'):
        """Initialize the model base class"""
        self.model_name = model_name
        self.model = None
        self.history = None
        self.log_dir = os.path.join(log_dir, model_name)
        os.makedirs(self.log_dir, exist_ok=True)

    @abstractmethod
    def build(self):
        pass

    def compile(self, learning_rate=0.001, metrics=None):
        """Compile the model with optimizer and loss function"""

        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        if metrics is None:
            metrics = ['accuracy']
        
        logger.info(f"Compiling model with learning rate: {learning_rate}")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=metrics
        )
        
        logger.info("Model compiled successfully")
        return self
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        self.model.summary()
        return self
    
    
    def get_callbacks(self, patience=5):
        """Get callbacks for training"""
        logger.info(f"Setting up callbacks with patience={patience}")
        
        # Model checkpoint to save best model
        checkpoint_path = os.path.join(self.log_dir, 'best_model.keras')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # TensorBoard for visualizing training
        tensorboard = TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True
        )
        
        return [checkpoint, early_stopping, tensorboard]    
    

    def fit(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=10, 
            patience=5, class_weights=None):
        """Train the model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size
            epochs: Number of epochs
            patience: Patience for early stopping
            class_weights: Weights for imbalanced classes
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not compiled. Call build() and compile() first.")
        
        logger.info(f"Starting training with batch_size={batch_size}, epochs={epochs}")
        
        # Log class distribution
        train_class_counts = {i: int(sum(y_train == i)) for i in set(y_train)}
        val_class_counts = {i: int(sum(y_val == i)) for i in set(y_val)}
        logger.info(f"Training class distribution: {train_class_counts}")
        logger.info(f"Validation class distribution: {val_class_counts}")
        
        if class_weights:
            logger.info(f"Using class weights: {class_weights}")
        
        callbacks = self.get_callbacks(patience=patience)
        
        # Log shapes for debugging
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"y_val shape: {y_val.shape}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        logger.info("Evaluating model on test data")
        
        # Log test class distribution
        test_class_counts = {i: int(sum(y_test == i)) for i in set(y_test)}
        logger.info(f"Test class distribution: {test_class_counts}")
        
        results = self.model.evaluate(X_test, y_test, verbose=1)
        metrics = dict(zip(self.model.metrics_names, results))
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        logger.info(f"Making predictions on data with shape: {X.shape}")
        return self.model.predict(X)
    

    def save(self, filepath=None):
        """Save the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"{self.model_name}.keras")
        else:
            if not filepath.endswith('.keras'):
                filepath = filepath.rsplit('.', 1)[0] + '.keras'
        
        logger.info(f"Saving model to {filepath}")
        self.model.save(filepath)
        return filepath
    
    def load(self, filepath):
        """Load the model"""
        logger.info(f"Loading model from {filepath}")
        self.model = tf.keras.models.load_model(filepath)
        return self
    

    def log_to_mlflow(self, run_id=None):
        """Log model and metrics to MLflow
        
        Args:
            run_id: MLflow run ID to log to
        """
        logger.info("Logging model to MLflow")
        
        try:
            # Check if a run is already active
            active_run = mlflow.active_run()
            
            # If a run is active and has the same ID, use it directly
            if active_run and active_run.info.run_id == run_id:
                logger.info(f"Using existing active MLflow run: {run_id}")
                self._log_model_to_mlflow()
            # If there's an active run with a different ID or no run_id was provided
            elif active_run:
                logger.info(f"Different MLflow run already active: {active_run.info.run_id}")
                # Use the active run directly
                self._log_model_to_mlflow()
            # If no run is active, start a new one
            else:
                logger.info(f"Starting new MLflow run with ID: {run_id}")
                with mlflow.start_run(run_id=run_id):
                    self._log_model_to_mlflow()
        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")
            # Don't re-raise to allow the pipeline to continue
            
        return self

    def _log_model_to_mlflow(self):
        """Helper method to log model artifacts to MLflow in the current run"""
        # Log model architecture
        model_config = {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_layers': len(self.model.layers),
            'num_parameters': self.model.count_params()
        }
        mlflow.log_params(model_config)
        
        # Log metrics if available
        if hasattr(self, 'history') and self.history is not None:
            for metric, values in self.history.history.items():
                # Log final epoch metrics
                mlflow.log_metric(f"final_{metric}", values[-1])
                
                # Log best metrics
                if 'val_' in metric:
                    best_epoch = np.argmax(values) if 'accuracy' in metric else np.argmin(values)
                    mlflow.log_metric(f"best_{metric}", values[best_epoch])
                    mlflow.log_metric(f"best_{metric}_epoch", best_epoch + 1)
        
        # Log model as an artifact
        mlflow.tensorflow.log_model(
            self.model,
            "model",
            registered_model_name=self.model_name
        )
        
        # Log model summary
        try:
            from io import StringIO
            summary_io = StringIO()
            self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
            mlflow.log_text(summary_io.getvalue(), "model_summary.txt")
        except Exception as e:
            logger.warning(f"Failed to log model summary: {str(e)}")