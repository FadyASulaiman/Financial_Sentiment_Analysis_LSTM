import os
from typing import List
import yaml
import pandas as pd
import numpy as np
import datetime
import hashlib
import pickle

import mlflow
from sklearn.pipeline import Pipeline

from src.config.config_loader import load_config
from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.feature_extractors.financial_entity import FinancialEntityExtractor
from src.feature_extractors.financial_event import FinancialEventClassifier
from src.feature_extractors.financial_ngram import FinancialNGramExtractor
from src.feature_extractors.growth_decline import GrowthDeclineQuantifier
from src.feature_extractors.industry_sector import IndustrySectorCategorizer
from src.feature_extractors.performace_indicator import FinancialPerformanceIndicatorExtractor
from src.feature_extractors.relative_change import RelativeChangeExtractor

from src.utils.feat_eng_pipeline_logger import logger



class FeatureEngineeringPipeline:
    """Pipeline orchestrator for feature engineering"""
    
    def __init__(self, config_path: str = None):
        """Initialize the pipeline with configuration"""
        
        self.config = load_config(config_path)

        if 'version' in self.config:
            self.version = self.config['version']
        
        # Set random seed
        np.random.seed(self.config.get('random_seed', 42))
        
        # Create transformers
        self.transformers = self._create_transformers()
        
        # Create sklearn pipeline
        self.pipeline = Pipeline([
            (transformer.name, transformer) for transformer in self.transformers
        ])
        
    def _create_transformers(self) -> List[FeatureExtractorBase]:
        """Create transformer objects based on configuration"""
        return [
            RelativeChangeExtractor(self.config),
            FinancialNGramExtractor(self.config),
            FinancialEntityExtractor(self.config),
            FinancialPerformanceIndicatorExtractor(self.config),
            GrowthDeclineQuantifier(self.config),
            IndustrySectorCategorizer(self.config),
            FinancialEventClassifier(self.config)
        ]
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the pipeline to data and transform it"""
        try:
            logger.info("Starting feature engineering process...")
            
            # Start MLflow run
            mlflow.start_run(run_name=f"feature_engineering_v{self.version}")
            
            self._log_parameters()
            
            # Fit and transform data
            start_time = datetime.datetime.now()
            transformed_data = self.pipeline.fit_transform(data)
            end_time = datetime.datetime.now()
            
            # Add original target if present
            if 'sentiment' in data.columns:
                transformed_data['sentiment'] = data['sentiment']
            
            # Log metrics
            self._log_metrics(data, transformed_data, start_time, end_time)
            
            # Create a data hash for versioning
            data_hash = self._create_data_hash(data)
            mlflow.log_param("data_hash", data_hash)
            
            # End MLflow run
            mlflow.end_run()
            
            logger.info(f"Feature engineering completed. Generated {transformed_data.shape[1]} features.")
            
            return transformed_data
        
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {str(e)}")
            mlflow.end_run()
            raise
    
    def _log_parameters(self):
        """Log parameters to MLflow"""
        # Log version, random seed, config and transformer names
        mlflow.log_param("pipeline_version", self.version)
        
        mlflow.log_param("random_seed", self.config.get("random_seed", 42))
        

        feature_config = self.config.get("features", {})
        for key, value in feature_config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    mlflow.log_param(f"features.{key}.{subkey}", subvalue)
            else:
                mlflow.log_param(f"features.{key}", value)
        

        mlflow.log_param("transformers", [t.name for t in self.transformers])
    
    def _log_metrics(self, original_data: pd.DataFrame, transformed_data: pd.DataFrame, 
                    start_time: datetime.datetime, end_time: datetime.datetime):
        """Log metrics to MLflow"""
        # Log data shapes
        mlflow.log_metric("input_rows", original_data.shape[0])
        mlflow.log_metric("input_columns", original_data.shape[1])
        mlflow.log_metric("output_features", transformed_data.shape[1])
        
        # Log processing time
        processing_time = (end_time - start_time).total_seconds()
        mlflow.log_metric("processing_time_seconds", processing_time)
        
        # Log feature group counts
        feature_groups = {}
        for transformer in self.transformers:
            feature_names = transformer.get_feature_names()
            feature_groups[transformer.name] = len(feature_names)
            mlflow.log_metric(f"feature_count_{transformer.name}", len(feature_names))
        
        # Log class distribution
        if 'sentiment' in original_data.columns:
            class_distribution = original_data['sentiment'].value_counts().to_dict()
            for class_name, count in class_distribution.items():
                mlflow.log_metric(f"class_count_{class_name}", count)
    
    def _create_data_hash(self, data: pd.DataFrame) -> str:
        """Create a hash of the data for versioning"""
        data_str = data.to_string().encode('utf-8')
        return hashlib.md5(data_str).hexdigest()
    
    def save_pipeline(self, filepath: str):
        """Save the fitted pipeline to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as file:
                pickle.dump(self.pipeline, file)
            logger.info(f"Pipeline saved to {filepath}")
            
            # Log the pipeline as an MLflow artifact
            mlflow.log_artifact(filepath)
            
        except Exception as e:
            logger.error(f"Error saving pipeline: {str(e)}")
    
    def save_config(self, filepath: str):
        """Save the configuration to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as file:
                yaml.dump(self.config, file)
            logger.info(f"Configuration saved to {filepath}")
            
            # Log the config as an MLflow artifact
            mlflow.log_artifact(filepath)
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")