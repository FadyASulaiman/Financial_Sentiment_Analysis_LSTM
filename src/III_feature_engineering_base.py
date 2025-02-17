from abc import ABC, abstractmethod
import re
import time
from typing import List, Union, Optional
import numpy as np
import pandas as pd
import scipy.sparse

import logging
from dataclasses import dataclass


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
        start_time = time.time()
        self.fit(X, y)
        features = self.transform(X)
        end_time = time.time()

        self.stats = self._calculate_stats(features)
        self.stats.extraction_time = end_time - start_time
        self._is_fitted = True
        return features


    def _validate_input(self, X: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate input DataFrame"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(X)}")

        missing_cols = set(required_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    def _calculate_stats(self, features: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]) -> FeatureStats:
        """Calculate feature statistics"""
        if isinstance(features, pd.DataFrame):
            n_samples, n_features = features.shape
            feature_names = features.columns.tolist()
            memory_usage = features.memory_usage(deep=True).sum() / 1024**2  # MB
            sparsity = features.isnull().sum().sum() / (n_samples * n_features) if n_samples * n_features > 0 else 0

        elif isinstance(features, np.ndarray):
            n_samples, n_features = features.shape if features.ndim > 1 else (features.shape[0],1)
            feature_names = [f"feature_{i}" for i in range(n_features)]
            memory_usage = features.nbytes / 1024**2  # MB
            sparsity = np.count_nonzero(features == 0) / (n_samples * n_features) if n_samples * n_features > 0 else 0

        elif isinstance(features, scipy.sparse.spmatrix):
            n_samples, n_features = features.shape
            feature_names = [f"feature_{i}" for i in range(n_features)]
            memory_usage = features.data.nbytes / 1024**2  # MB
            sparsity = 1 - (features.nnz / (n_samples * n_features)) if n_samples * n_features > 0 else 0

        else: 
            raise TypeError(f"Unsupported feature type: {type(features)}")


        return FeatureStats(
            n_samples=n_samples,
            n_features=n_features,
            feature_names=feature_names,
            memory_usage=memory_usage,
            extraction_time=0.0,  # Placeholder, updated in fit_transform
            sparsity=sparsity
        )


class TextPreprocessor:
    """Text preprocessing utilities for financial texts."""

    def __init__(self):
        self.financial_patterns = {
            'currency': r'[\$|£|€]',  # Match any of the currency symbols
            'numbers': r'\b\d+(?:,\d+)*(?:\.\d+)?\b(?!\s*(?:million|billion|trillion|percent|%))', # Match numbers with optional commas and decimals
            'percentages': r'\b\d+(?:,\d+)*(?:\.\d+)?%\b', # Match percentages with optional commas and decimals
            'quantities': r'\b(million|mn|billion|bn|trillion|tn)\b'
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize financial text."""

        text = text.lower().strip()
        text = self._normalize_financial_terms(text)
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace

        return text

    def _normalize_financial_terms(self, text: str) -> str:
        """Normalize financial terms and numbers in text."""

        text = re.sub(self.financial_patterns['currency'], 'currency_symbol', text)
        text = re.sub(r'\b(million|mn)\b', 'million', text)
        text = re.sub(r'\b(billion|bn)\b', 'billion', text)
        text = re.sub(r'\b(trillion|tn)\b', 'trillion', text)
        text = re.sub(self.financial_patterns['percentages'], 'percent_value', text)
        text = re.sub(self.financial_patterns['numbers'], 'number_value', text)

        return text