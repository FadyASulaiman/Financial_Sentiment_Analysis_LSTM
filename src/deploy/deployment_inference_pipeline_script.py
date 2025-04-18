import pickle

import numpy as np
import pandas as pd

from src.utils.loggers.prod_logger import logger

import keras

# Define the inference function at module level, not nested inside another function
def production_inference_pipeline(input_text, pipeline_paths, text_column='Sentence'):
    """
    Production inference pipeline that processes a single text string
    through all preprocessing steps and returns model predictions
    
    Args:
        input_text: A string containing the raw text input
        pipeline_paths: Dictionary containing paths to all pipeline components
        text_column: Column name to use in the temporary DataFrame
        
    Returns:
        Model predictions
    """
    try:
        fe_pipeline_path = pipeline_paths['fe_pipeline_path']
        preprocessing_pipeline_path = pipeline_paths['preprocessing_pipeline_path']
        data_prep_pipeline_path = pipeline_paths['data_prep_pipeline_path']
        model_path = pipeline_paths['model_path']
        
        # Load pipelines
        with open(fe_pipeline_path, 'rb') as f:
            fe_pipeline = pickle.load(f)
            
        with open(preprocessing_pipeline_path, 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
            
        with open(data_prep_pipeline_path, 'rb') as f:
            data_prep_pipeline = pickle.load(f)
        
        # Load model
        model = keras.models.load_model(model_path)
        
        # Create a single-row DataFrame from the input string
        input_data = pd.DataFrame({text_column: [input_text]})

        # Step 1: Feature Engineering
        fe_data = fe_pipeline.transform(input_data)
        
        # Step 2: Text Preprocessing
        preprocessed_data = preprocessing_pipeline.transform(fe_data)
        print(preprocessed_data)
        print(type(preprocessed_data))


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
        return result

    except Exception as e:
        logger.error(f"Error in production inference: {str(e)}")
        raise