

from datetime import datetime
import hashlib
import os
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd

from src.deploy.inference_pipeline.prod_logger import logger

from src.preprocessors.data_preprocessors import (
    HTMLCleaner, URLRemover, PunctuationRemover, SpecialCharRemover,
    WhitespaceNormalizer, NumericNormalizer, DateTimeNormalizer,
    StockTickerReplacer, CurrencyReplacer, TextLowercaser,
    StopWordRemover
)

from src.feature_extractors.financial_entity import FinancialEntityExtractor

from src.feature_extractors.financial_event import FinancialEventClassifier
from src.feature_extractors.industry_sector import IndustrySectorClassifier

from src.preprocessors.data_prep.sequence_padder import SequencePadder
from src.preprocessors.data_prep.spacy_lemmatizer import SpacyLemmatizer

from src.preprocessors.data_prep.tokenizer import Tokenizer
from src.preprocessors.data_prep.vocab_builder import VocabularyBuilder

import os
import numpy as np
import pandas as pd
import datetime


class UnifiedPipeline:
    """
    A unified pipeline that combines feature engineering, data cleaning, and data preparation
    into a single processing pipeline for financial text data.
    """
    
    def __init__(self, 
                 config_path: str = None, 
                 max_sequence_length: int = 128, 
                 text_column: str = 'Sentence', 
                 label_column: str = 'Sentiment'):
        """
        Initialize the unified pipeline.
        
        Args:
            config_path: Path to configuration file for feature engineering
            max_sequence_length: Maximum sequence length for padding
            text_column: Name of the column containing text data
            label_column: Name of the column containing labels
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.max_sequence_length = max_sequence_length
        self.text_column = text_column
        self.label_column = label_column
        
        # Set random seed
        np.random.seed(self.config.get('random_seed', 42))
        
        # Initialize pipelines
        self.feature_engineering_pipeline = None
        self.data_cleaning_pipeline = None
        self.data_preparation_pipeline = None
        
        # Domain-specific stopwords for data cleaning
        self.domain_specific_stopwords = [
            'ltd', 'inc', 'corp', 'corporation', 'company', 'co', 'group',
            'plc', 'holdings', 'holding', 'international', 'technologies',
            'technology', 'solutions', 'services', 'system', 'systems',
            'quarter', 'year', 'month', 'day', 'week', 'today', 'yesterday',
            'tomorrow', 'said', 'announced', 'reported', 'according', 'statement'
        ]
        
        # Initialize pipelines
        self._initialize_pipelines()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _initialize_pipelines(self):
        """Initialize all component pipelines."""
        from sklearn.pipeline import Pipeline
        
        # Initialize Feature Engineering Pipeline
        self._initialize_feature_engineering()
        
        # Initialize Data Cleaning Pipeline
        self._initialize_data_cleaning()
        
        # Initialize Data Preparation Pipeline
        self._initialize_data_preparation()
    
    def _initialize_feature_engineering(self):
        """Initialize the feature engineering pipeline."""
        from sklearn.pipeline import Pipeline
        
        # This would typically create transformers based on your FeatureExtractorBase classes
        # For simplicity, assuming these are already implemented classes
        transformers = self._create_feature_transformers()
        
        self.feature_engineering_pipeline = Pipeline([
            (transformer.name, transformer) for transformer in transformers
        ])
        
        logger.info("Feature engineering pipeline initialized")
    
    def _create_feature_transformers(self) -> List:
        """Create transformer objects based on configuration."""
        # Assuming these classes are defined elsewhere in your codebase
        # Adjust imports as necessary
        return [
            FinancialEntityExtractor(self.config),
            FinancialEventClassifier(self.config),
            IndustrySectorClassifier(self.config),
        ]
    
    def _initialize_data_cleaning(self):
        """Initialize the data cleaning pipeline."""
        from sklearn.pipeline import Pipeline
        
        self.data_cleaning_pipeline = Pipeline([
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
            ('stop_word_remover', StopWordRemover(self.domain_specific_stopwords))
        ])
        
        logger.info("Data cleaning pipeline initialized")
    
    def _initialize_data_preparation(self):
        """Initialize the data preparation pipeline."""
        from sklearn.pipeline import Pipeline
        
        self.data_preparation_pipeline = Pipeline([
            ('tokenizer', Tokenizer(remove_stopwords=False)),
            ('lemmatizer', SpacyLemmatizer()),
            ('vocabulary', VocabularyBuilder(min_word_count=2, max_vocab_size=20000)),
            ('padder', SequencePadder(self.max_sequence_length))
        ])
        
        logger.info("Data preparation pipeline initialized")
    
    def process(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the input data through all three pipeline stages.
        
        Args:
            data: Input DataFrame containing text data
            
        Returns:
            Tuple of (X, y) arrays ready for model training/inference
        """
        try:
            logger.info("Starting unified pipeline processing")
            start_time = datetime.datetime.now()
            
            # Stage 1: Feature Engineering
            logger.info("Stage 1: Feature Engineering")
            fe_start_time = datetime.datetime.now()
            transformed_data = self.feature_engineering_pipeline.fit_transform(data)
            
            # Preserve target if present
            if self.label_column in data.columns:
                transformed_data[self.label_column] = data[self.label_column]
                
            fe_end_time = datetime.datetime.now()
            fe_duration = (fe_end_time - fe_start_time).total_seconds()
            logger.info(f"Feature engineering completed in {fe_duration:.2f} seconds. Generated {transformed_data.shape[1]} features.")
            
            # Stage 2: Data Cleaning
            logger.info("Stage 2: Data Cleaning")
            clean_start_time = datetime.datetime.now()
            
            # Extract sentences and labels
            sentences = transformed_data[self.text_column].copy()
            sentiments = transformed_data[self.label_column].copy()
            
            # Extract additional features if they exist
            additional_features = {}
            additional_feature_columns = ['Sector', 'Company', 'Event']
            for column in additional_feature_columns:
                if column in transformed_data.columns:
                    additional_features[column] = transformed_data[column].tolist()
            
            # Apply cleaning pipeline
            cleaned_sentences = self.data_cleaning_pipeline.fit_transform(sentences)
            
            # Create cleaned DataFrame
            cleaned_df = pd.DataFrame({
                self.text_column: cleaned_sentences,
                self.label_column: sentiments
            })
            
            # Add additional features to the DataFrame
            for feature_name, feature_values in additional_features.items():
                cleaned_df[feature_name] = feature_values
            
            clean_end_time = datetime.datetime.now()
            clean_duration = (clean_end_time - clean_start_time).total_seconds()
            logger.info(f"Data cleaning completed in {clean_duration:.2f} seconds")
            
            # Stage 3: Data Preparation
            logger.info("Stage 3: Data Preparation")
            prep_start_time = datetime.datetime.now()
            
            # Convert sentiment labels to numeric indices
            sentiment_mapping = {
                'negative': 0,
                'neutral': 1, 
                'positive': 2
            }
            
            labels = cleaned_df[self.label_column]
            
            # Check label type and convert if needed
            if labels.dtype == 'object' or labels.dtype.kind == 'U' or labels.dtype.kind == 'S':
                numeric_labels = labels.apply(lambda x: sentiment_mapping.get(str(x).lower(), 1))
                logger.info("Converted labels from strings to integers")
                y = np.array(numeric_labels)
            else:
                # Labels already numeric
                y = np.array(labels)
            
            # Apply data preparation pipeline
            X = self.data_preparation_pipeline.fit_transform(cleaned_df[self.text_column])
            
            # Convert to numpy array if not already
            X = np.array(X)
            
            prep_end_time = datetime.datetime.now()
            prep_duration = (prep_end_time - prep_start_time).total_seconds()
            logger.info(f"Data preparation completed in {prep_duration:.2f} seconds")
            
            # Log completion
            end_time = datetime.datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            logger.info(f"Unified pipeline completed in {total_duration:.2f} seconds")
            logger.info(f"Final output: X shape {X.shape}, y shape {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in unified pipeline: {str(e)}")
            raise
    
    def save(self, filepath: str):
        """
        Save the unified pipeline using joblib for production use.
        
        Args:
            filepath: Path where the pipeline will be saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create a dictionary with all three pipelines
            pipeline_dict = {
                'feature_engineering': self.feature_engineering_pipeline,
                'data_cleaning': self.data_cleaning_pipeline,
                'data_preparation': self.data_preparation_pipeline,
                'config': {
                    'max_sequence_length': self.max_sequence_length,
                    'text_column': self.text_column,
                    'label_column': self.label_column,
                    'domain_specific_stopwords': self.domain_specific_stopwords,
                }
            }
            
            # Save using joblib
            joblib.dump(pipeline_dict, filepath, compress=3)
            logger.info(f"Unified pipeline saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving unified pipeline: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a saved unified pipeline.
        
        Args:
            filepath: Path to the saved pipeline
            
        Returns:
            Loaded UnifiedPipeline instance
        """
        try:
            # Load the pipeline dictionary
            pipeline_dict = joblib.load(filepath)
            
            # Create a new instance
            instance = cls()
            
            # Restore pipelines
            instance.feature_engineering_pipeline = pipeline_dict['feature_engineering']
            instance.data_cleaning_pipeline = pipeline_dict['data_cleaning']
            instance.data_preparation_pipeline = pipeline_dict['data_preparation']
            
            # Restore config
            config = pipeline_dict['config']
            instance.max_sequence_length = config['max_sequence_length']
            instance.text_column = config['text_column']
            instance.label_column = config['label_column']
            instance.domain_specific_stopwords = config['domain_specific_stopwords']
            
            logger.info(f"Unified pipeline loaded from {filepath}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading unified pipeline: {str(e)}")
            raise

