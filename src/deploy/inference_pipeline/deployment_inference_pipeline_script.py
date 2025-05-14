from datetime import datetime
import os


import joblib
import numpy as np
import pandas as pd

from .prod_logger import logger

import keras

class ProductionPredictor:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.fe_pipeline_path = os.path.join(script_dir, "artifacts", "feature_pipeline.pkl")
        self.preprocessing_pipeline_path = os.path.join(script_dir, "artifacts", "cleaning_pipeline.pkl")
        self.data_prep_pipeline_path = os.path.join(script_dir, "artifacts", "data_prep_pipeline.pkl") 
        self.model_path = os.path.join(script_dir, "artifacts", "lstm_sentiment_80_acc.keras")
        self.unified_pipeline_path = os.path.join(script_dir, "artifacts", "unified_pipeline.joblib")
        
        self.unified_pipeline = joblib.load(self.unified_pipeline_path)
        self.model = keras.models.load_model(self.model_path)

    def production_inference_pipeline(self, input_text, text_column='Sentence'):
        """
        Production inference pipeline that processes a single text string
        through all preprocessing steps and returns model predictions
        """
        try:
            # Track start time
            start_time = datetime.now()


            if self.unified_pipeline is None:
                # Load unified pipeline
                logger.info(f"Loading unified pipeline from {self.unified_pipeline_path}")
                self.unified_pipeline = joblib.load(self.unified_pipeline_path)
                
            if self.model is None:
                # Load model
                logger.info(f"Loading model from {self.model_path}")
                self.model = keras.models.load_model(self.model_path)

            # Track loading end time and duration
            end_time_first = datetime.now()
            loading_duration = (end_time_first - start_time).total_seconds()
            logger.info(f"Loading completed in {loading_duration:.2f} seconds")
            
            # Start preprocessing
            input_data = pd.DataFrame({text_column: [input_text]})

            # Process through feature engineering pipeline
            logger.info("Applying feature engineering transformation")
            fe_data = self.unified_pipeline['feature_engineering'].transform(input_data)
            
            # Ensure the text column is preserved
            if text_column not in fe_data.columns:
                fe_data[text_column] = input_data[text_column]
            
            # Process through data cleaning pipeline
            logger.info("Applying data cleaning transformation")
            cleaned_sentences = self.unified_pipeline['data_cleaning'].transform(fe_data[text_column])
            
            # Create DataFrame with cleaned text
            cleaned_data = pd.DataFrame({text_column: cleaned_sentences})
            
            # Process through data preparation pipeline
            logger.info("Applying data preparation transformation")
            X = self.unified_pipeline['data_preparation'].transform(cleaned_data[text_column])
            
            # Ensure X is in the right shape for model prediction
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)  # Add batch dimension if needed
            
            # Model Prediction
            logger.info("Running model prediction")
            predictions = self.model.predict(X)
            
            # Convert predictions to human-readable format
            sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            pred_class = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][pred_class])
            result = {
                'sentiment': sentiment_mapping.get(pred_class, 'unknown'),
                'confidence': confidence,
                'raw_predictions': predictions[0].tolist()
            }
            
            logger.info(result)

            # Track end time and duration
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            logger.info(f"Inference completed in {processing_duration:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error in production inference: {str(e)}")
            raise