from abc import ABC, abstractmethod
import re
from typing import List, Dict, Union, Optional, Any
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import logging
from dataclasses import dataclass
from pathlib import Path
import os
from tqdm import tqdm

@dataclass
class FeatureStats:
    """Statistics for feature extraction process"""
    n_samples: int
    n_features: int
    feature_names: List[str]
    memory_usage: float
    extraction_time: float
    sparsity: float

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats: Optional[FeatureStats] = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseFeatureExtractor':
        """Fit the feature extractor"""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]:
        """Transform the input data"""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]:
        """Fit and transform the data"""
        return self.fit(X, y).transform(X)

    def _validate_input(self, X: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate input DataFrame"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(X)}")
        
        missing_cols = [col for col in required_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _calculate_stats(self, features: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]) -> FeatureStats:
        """Calculate feature statistics"""
        if isinstance(features, pd.DataFrame):
            n_samples, n_features = features.shape
            feature_names = features.columns.tolist()
            memory_usage = features.memory_usage(deep=True).sum() / 1024**2  # MB
            sparsity = features.isnull().sum().sum() / (n_samples * n_features)
        else:
            n_samples = features.shape[0]
            n_features = features.shape[1] if len(features.shape) > 1 else 1
            feature_names = [f"feature_{i}" for i in range(n_features)]
            
            if isinstance(features, scipy.sparse.spmatrix):
                memory_usage = features.data.nbytes / 1024**2  # MB
                sparsity = 1 - (features.nnz / (n_samples * n_features))
            else:
                memory_usage = features.nbytes / 1024**2  # MB
                sparsity = np.count_nonzero(features == 0) / (n_samples * n_features)

        return FeatureStats(
            n_samples=n_samples,
            n_features=n_features,
            feature_names=feature_names,
            memory_usage=memory_usage,
            extraction_time=0.0,  # Will be set during extraction
            sparsity=sparsity
        )

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self):
        self.financial_patterns = {
            'currency': r'(\$|£|€)',
            'numbers': r'\b\d+(?!\s*(?:million|billion|trillion|percent|%))\b',
            'percentages': r'\b\d+(\.\d+)?%\b',
            'quantities': r'\b(million|mn|billion|bn|trillion|tn)\b'
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Basic cleaning
        text = text.lower().strip()
        
        # Normalize financial terms
        text = self._normalize_financial_terms(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def _normalize_financial_terms(self, text: str) -> str:
        """Normalize financial terms and numbers"""
        # Replace currency symbols
        text = re.sub(self.financial_patterns['currency'], 'currency_symbol', text)
        
        # Normalize quantity terms
        text = re.sub(r'\b(million|mn)\b', 'million', text)
        text = re.sub(r'\b(billion|bn)\b', 'billion', text)
        text = re.sub(r'\b(trillion|tn)\b', 'trillion', text)
        
        # Normalize percentages
        text = re.sub(self.financial_patterns['percentages'], 'percent_value', text)
        
        # Handle numbers with special cases
        def replace_number(match):
            num = match.group(0)
            try:
                if float(num) > 1000000:
                    return 'large_number'
                return 'number'
            except ValueError:
                return num

        text = re.sub(self.financial_patterns['numbers'], replace_number, text)
        
        return text