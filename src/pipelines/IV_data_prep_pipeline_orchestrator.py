import os
import pickle
import pandas as pd
import mlflow
from datetime import datetime
from sklearn.pipeline import Pipeline
import numpy as np

from src.preprocessors.data_prep.sequence_padder import SequencePadder
from src.preprocessors.data_prep.spacy_lemmatizer import SpacyLemmatizer

from src.preprocessors.data_prep.tokenizer import Tokenizer
from src.preprocessors.data_prep.vocab_builder import VocabularyBuilder
from src.utils.data_prep_pipeline_logger import logger


class DataPreparationOrchestrator:
    """Main Data Preparation orchestrator"""
    
    def __init__(self, text_column='Sentence', label_column='Sentiment', max_sequence_length=128):
        self.max_sequence_length = max_sequence_length
        self.text_column = text_column
        self.label_column = label_column
        self.data_prep_pipeline = None
        self.mlflow_run_id = None
        
        # Initialize MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('data_prep')
    
    def build_pipeline(self):
        """Build the preprocessing pipeline"""
        logger.info("Building preprocessing pipeline")
        
        self.data_prep_pipeline = Pipeline([
            ('tokenizer', Tokenizer(remove_stopwords=False)),
            ('lemmatizer', SpacyLemmatizer()),
            ('vocabulary', VocabularyBuilder(min_word_count=2, max_vocab_size=20000)),
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
                mlflow.log_param("text_column", self.text_column)
                mlflow.log_param("label_column", self.label_column)
                
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
                mlflow.log_param("dataset_columns", list(df.columns))
                
                # Determine output directory
                if output_dir is None and data_path is not None:
                    output_dir = os.path.dirname(data_path)
                elif output_dir is None:
                    output_dir = "."
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Build pipeline if not already built
                if self.data_prep_pipeline is None:
                    self.build_pipeline()
                
                # Extract text and labels
                sentences = df[self.text_column].copy()
                labels = df[self.label_column].copy()
                
                # Keep other columns for later use
                other_columns = {col: df[col] for col in df.columns 
                                if col not in [self.text_column, self.label_column]}
                
                # Track start time
                start_time = datetime.now()
                
                # Apply preprocessing pipeline
                processed_sentences = self.data_prep_pipeline.fit_transform(sentences, labels)

                # Track end time and duration
                end_time = datetime.now()
                processing_duration = (end_time - start_time).total_seconds()
                mlflow.log_metric("data_prep_duration_seconds", processing_duration)
                logger.info(f"Data Preparation completed in {processing_duration:.2f} seconds")
                
                # Create a new DataFrame with the original data
                processed_df = df.copy()

                X = processed_sentences  # This is already a 2D numpy array
                y = np.array(labels)


                # Save processed data as numpy arrays for LSTM model
                np_output_path = os.path.join(output_dir, 'processed_arrays.npz')
                np.savez(np_output_path, X=X, y=y)
                logger.info(f"Processed arrays saved to {np_output_path}")


                # Serialize the pipeline
                pipeline_path = os.path.join(output_dir, 'data_prep_pipeline.pkl')
                with open(pipeline_path, 'wb') as f:
                    pickle.dump(self.data_prep_pipeline, f)
                logger.info(f"Data prepping pipeline serialized to {pipeline_path}")
                
                # Log artifacts
                mlflow.log_artifact(np_output_path)
                mlflow.log_artifact(pipeline_path)
                
                return processed_df, X, y
                
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def _load_data(self, data_path):
        """Load data from CSV file"""
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Validate required columns
            if self.text_column not in df.columns or self.label_column not in df.columns:
                raise ValueError(f"CSV must contain '{self.text_column}' and '{self.label_column}' columns")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise