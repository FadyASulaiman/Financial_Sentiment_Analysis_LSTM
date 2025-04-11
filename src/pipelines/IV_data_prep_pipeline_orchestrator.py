import os
import pickle
import mlflow
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from src.preprocessors.data_prep.finbert_tokenizer import FinBERTTokenizer
from src.preprocessors.data_prep.sequence_padder import SequencePadder
from src.preprocessors.data_prep.spacy_lemmatizer import SpacyLemmatizer
from src.utils.preprocessing_logger import logger


class DataPreparationOrchestrator:
    """Main Data Preparation orchestrator"""
    
    def __init__(self, max_sequence_length=128):
        self.max_sequence_length = max_sequence_length

        self.preprocessing_pipeline = None
        self.mlflow_run_id = None
        
        # Initialize MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('data_prep')
    
    def build_pipeline(self):
        """Build the preprocessing pipeline"""
        logger.info("Building preprocessing pipeline")
        
        self.data_prep_pipeline = Pipeline([
            ('tokenizer', FinBERTTokenizer()),
            ('lemmatizer', SpacyLemmatizer()),
            ('padder', SequencePadder(self.max_sequence_length))
        ])
        
        return self.data_prep_pipeline
    
    def preprocess(self, data_path=None, data_df=None, output_dir=None):
        """Preprocess the data and track with MLflow"""
        try:
            # Start MLflow run
            with mlflow.start_run() as run:
                self.mlflow_run_id = run.info.run_id
                logger.info(f"Started MLflow run with ID: {self.mlflow_run_id}")
                
                # Log parameters
                mlflow.log_param("max_sequence_length", self.max_sequence_length)
                
                # Load data
                if data_df is not None:
                    df = data_df.copy()
                elif data_path is not None:
                    df = self._load_data(data_path)
                    mlflow.log_param("data_path", data_path)
                else:
                    raise ValueError("Either data_path or data_df must be provided")
                
                # Log dataset info
                mlflow.log_param("dataset_size", len(df))
                
                # Determine output directory
                if output_dir is None and data_path is not None:
                    output_dir = os.path.dirname(data_path)
                elif output_dir is None:
                    output_dir = "."
                
                # Build pipeline if not already built
                if self.data_prep_pipeline is None:
                    self.build_pipeline()
                
                # Extract sentences and sentiments
                sentences = df['Sentence'].copy()
                sentiments = df['Sentiment'].copy()
                
                # Track start time
                start_time = datetime.now()
                
                # Apply preprocessing pipeline
                processed_sentences = self.data_prep_pipeline.fit_transform(sentences, sentiments)


                # Track end time and duration
                end_time = datetime.now()
                processing_duration = (end_time - start_time).total_seconds()
                mlflow.log_metric("processing_duration_seconds", processing_duration)
                logger.info(f"Preprocessing completed in {processing_duration:.2f} seconds")
                
                # Create a new DataFrame with the processed data
                processed_df = pd.DataFrame({
                    'Sentence': processed_sentences,
                    'Sentiment': sentiments
                })


                
                # Save the processed data
                output_path = os.path.join(output_dir, 'processed_data.csv')
                processed_df.to_csv(output_path, index=False)
                logger.info(f"Processed data saved to {output_path}")
                
                # Log metrics
                mlflow.log_metric("processed_data_size", len(processed_df))
                
                # Serialize the pipeline
                pipeline_path = os.path.join(output_dir, 'data_prep_pipeline.pkl')
                with open(pipeline_path, 'wb') as f:
                    pickle.dump(self.preprocessing_pipeline, f)
                logger.info(f"Data prepping pipeline serialized to {pipeline_path}")
                
                # Log artifacts
                mlflow.log_artifact(output_path)
                mlflow.log_artifact(pipeline_path)
                
                return processed_df
                
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def _load_data(self, data_path):
        """Load data from CSV file"""
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Validate required columns
            if 'Sentence' not in df.columns or 'Sentiment' not in df.columns:
                raise ValueError("CSV must contain 'Sentence' and 'Sentiment' columns")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_pipeline(self, pipeline_path):
        """Load a serialized preprocessing pipeline"""
        try:
            logger.info(f"Loading preprocessing pipeline from {pipeline_path}")
            with open(pipeline_path, 'rb') as f:
                self.preprocessing_pipeline = pickle.load(f)
            return self.preprocessing_pipeline
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            raise