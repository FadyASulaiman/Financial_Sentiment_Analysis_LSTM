from datetime import datetime
import os
import pickle

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

    def production_inference_pipeline(self, input_text, text_column='Sentence'):
        """
        Production inference pipeline that processes a single text string
        through all preprocessing steps and returns model predictions

        """
        try:
            # Track start time
            start_time = datetime.now()

            # Load pipelines & model
            with open(self.fe_pipeline_path, 'rb') as f:
                fe_pipeline = pickle.load(f)
                
            with open(self.preprocessing_pipeline_path, 'rb') as f:
                preprocessing_pipeline = pickle.load(f)
                
            with open(self.data_prep_pipeline_path, 'rb') as f:
                data_prep_pipeline = pickle.load(f)
            
            model = keras.models.load_model(self.model_path)

            # Track un-pickling end time and duration
            end_time_first = datetime.now()
            processing_duration_f = (end_time_first - start_time).total_seconds()
            logger.info(f"Unpickling completed in {processing_duration_f:.2f} seconds")
            
            # Start preprocessing
            input_data = pd.DataFrame({text_column: [input_text]})

            # Step 1: Feature Engineering
            fe_data = fe_pipeline.transform(input_data)
            
            # Step 2: Text Preprocessing
            preprocessed_data = preprocessing_pipeline.transform(fe_data)

            # Step 3: Data Preparation
            X = data_prep_pipeline.transform(preprocessed_data["Sentence"])
            
            # Step 4: Model Prediction
            predictions = model.predict(X)
            
            # Convert predictions to human-readable format
            sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            pred_class = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][pred_class])
            result = {
                'sentiment': sentiment_mapping.get(pred_class, 'unknown'),
                'confidence': confidence,
                'raw_predictions': predictions[0].tolist()
            }
            print(result)
            # Track end time and duration
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            logger.info(f"Inference completed in {processing_duration:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error in production inference: {str(e)}")
            raise