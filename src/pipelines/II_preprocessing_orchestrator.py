import os
import pickle
import mlflow
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from src.preprocessors.data_preprocessors.minority_oversampler import SentimentBalancer
from src.utils.loggers.preprocessing_logger import logger
from src.preprocessors.data_preprocessors import (
    HTMLCleaner, URLRemover, PunctuationRemover, SpecialCharRemover,
    WhitespaceNormalizer, NumericNormalizer, DateTimeNormalizer,
    StockTickerReplacer, CurrencyReplacer, TextLowercaser,
    StopWordRemover
)

class FinanceTextPreprocessingOrchestrator:
    """Main preprocessing orchestrator"""
    
    def __init__(self, max_sequence_length=128, domain_specific_stopwords=None):
        self.max_sequence_length = max_sequence_length
        self.domain_specific_stopwords = domain_specific_stopwords or [
            'ltd', 'inc', 'corp', 'corporation', 'company', 'co', 'group',
            'plc', 'holdings', 'holding', 'international', 'technologies',
            'technology', 'solutions', 'services', 'system', 'systems',
            'quarter', 'year', 'month', 'day', 'week', 'today', 'yesterday',
            'tomorrow', 'said', 'announced', 'reported', 'according', 'statement'
        ]
        self.preprocessing_pipeline = None
        self.mlflow_run_id = None
        
        # Initialize MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('finance_text_preprocessing')
    
    def build_pipeline(self):
        """Build the preprocessing pipeline"""
        logger.info("Building preprocessing pipeline")
        
        self.preprocessing_pipeline = Pipeline([
            ('html_cleaner', HTMLCleaner()),
            ('url_remover', URLRemover()),
            ('currency_replacer', CurrencyReplacer()),
            ('stock_ticker_replacer', StockTickerReplacer()),
            ('date_time_normalizer', DateTimeNormalizer()),
            ('numeric_normalizer', NumericNormalizer()),
            ('punctuation_remover', PunctuationRemover()),
            ('special_char_remover', SpecialCharRemover()),
            ('whitespace_normalizer', WhitespaceNormalizer()),
            ('lowercaser', TextLowercaser()),
            ('stop_word_remover', StopWordRemover(self.domain_specific_stopwords)),
            ('minority_oversampler' , SentimentBalancer()) 
        ])
        
        return self.preprocessing_pipeline
    def preprocess(self, data_path=None, data_df=None, output_dir=None):
        """Preprocess the data and track with MLflow"""
        try:
            # Start MLflow run
            with mlflow.start_run() as run:
                self.mlflow_run_id = run.info.run_id
                logger.info(f"Started MLflow run with ID: {self.mlflow_run_id}")
                
                # Log parameters
                mlflow.log_param("max_sequence_length", self.max_sequence_length)
                mlflow.log_param("domain_specific_stopwords", self.domain_specific_stopwords)
                
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
                mlflow.log_param("sentiment_classes", df['Sentiment'].unique().tolist())
                
                # Determine output directory
                if output_dir is None and data_path is not None:
                    output_dir = os.path.dirname(data_path)
                elif output_dir is None:
                    output_dir = "."
                
                # Build pipeline if not already built
                if self.preprocessing_pipeline is None:
                    self.build_pipeline()
                
                # Extract sentences and sentiments
                sentences = df['Sentence'].copy()
                sentiments = df['Sentiment'].copy()
                
                # Extract additional features if they exist
                additional_features = {}
                additional_feature_columns = ['Sector', 'Company', 'Event']

                for column in additional_feature_columns:
                    if column in df.columns:
                        # Always store as lists to avoid numpy array extend() issues
                        additional_features[column] = df[column].tolist()
                        logger.info(f"Extracted {len(df[column])} values for additional feature '{column}'")
                # Track start time
                start_time = datetime.now()
                
                # Set additional features for the balancer
                if additional_features:
                    logger.info("Setting additional features for the balancer")
                    self.preprocessing_pipeline.named_steps['minority_oversampler'].set_additional_features(additional_features)
                
                # Apply preprocessing pipeline
                processed_sentences = self.preprocessing_pipeline.fit_transform(sentences, sentiments)
                logger.info(f"Generated {len(processed_sentences)} processed sentences")

                # Get balanced sentiments
                balancer = self.preprocessing_pipeline.named_steps['minority_oversampler']
                balanced_sentiments = balancer.get_balanced_y()
                logger.info(f"Retrieved {len(balanced_sentiments)} balanced sentiment labels")
                
                # Create a new DataFrame with the processed data
                processed_df = pd.DataFrame({
                    'Sentence': processed_sentences,
                    'Sentiment': balanced_sentiments
                })
                
                # Add balanced additional features to the DataFrame
                balanced_features = balancer.get_balanced_additional_features()
                if balanced_features:
                    logger.info(f"Adding balanced features to final DataFrame: {list(balanced_features.keys())}")
                    for feature_name, feature_values in balanced_features.items():
                        processed_df[feature_name] = feature_values
                        logger.info(f"Added {len(feature_values)} values for balanced feature '{feature_name}'")
                else:
                    logger.warning("No balanced additional features returned by the balancer")
                
                # Track end time and duration
                end_time = datetime.now()
                processing_duration = (end_time - start_time).total_seconds()
                mlflow.log_metric("processing_duration_seconds", processing_duration)
                logger.info(f"Preprocessing completed in {processing_duration:.2f} seconds")
                
                # Save Preprocessed data
                output_path = os.path.join(output_dir, 'preprocessed_FE_data.csv')
                processed_df.to_csv(output_path, index=False)
                logger.info(f"Processed data saved to {output_path} with columns: {processed_df.columns.tolist()}")
                
                # Log metrics
                mlflow.log_metric("processed_data_size", len(processed_df))
                
                # Serialize the pipeline
                pipeline_path = os.path.join(output_dir, 'preprocessing_pipeline.pkl')
                with open(pipeline_path, 'wb') as f:
                    pickle.dump(self.preprocessing_pipeline, f)
                logger.info(f"Preprocessing pipeline serialized to {pipeline_path}")
                
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