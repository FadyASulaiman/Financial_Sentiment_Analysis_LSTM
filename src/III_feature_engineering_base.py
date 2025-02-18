from abc import ABC, abstractmethod
import re
import time
from typing import List, Union, Optional
import numpy as np
import pandas as pd
import scipy.sparse

import logging
from dataclasses import dataclass

import spacy

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
    """Abstract base class for feature extractors with enhanced validation"""

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
        self.stats = self._calculate_stats(features)
        self.stats.extraction_time = time.time() - start_time
        self._is_fitted = True
        return features

    def _validate_input(self, X: pd.DataFrame, required_columns: List[str]) -> None:
        """Enhanced input validation with NaN checks"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(X)}")

        missing_cols = set(required_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        for col in required_columns:
            if X[col].isnull().any():
                raise ValueError(f"Column {col} contains NaN values")
            
    def _calculate_stats(self, features: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]) -> FeatureStats:
        """Universal statistics calculation for all feature types"""
        if isinstance(features, pd.DataFrame):
            n_samples, n_features = features.shape
            feature_names = features.columns.tolist()
            memory_usage = features.memory_usage(deep=True).sum() / 1024**2  # MB
            sparsity = features.isnull().sum().sum() / (n_samples * n_features) if n_samples * n_features > 0 else 0
        elif isinstance(features, np.ndarray):
            n_samples, n_features = features.shape if features.ndim > 1 else (features.shape[0], 1)
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
            extraction_time=0.0,  # Placeholder
            sparsity=sparsity
        )
    

class TextPreprocessor:
    """Enhanced financial text preprocessing with lemmatization"""

    FINANCIAL_PATTERNS = {
        'currency_amount': r'(?:[\$£€]\s*)?\d+(?:,\d+)*(?:\.\d+)?\b',
        'financial_quantity': r'\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:million|mn|billion|bn|trillion|tn)\b',
        'percentage': r'\b\d+(?:\.\d+)?%\b'
    }

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])
        self.lemmatizer = self.nlp.get_pipe("lemmatizer")

    def clean_text(self, text: str) -> str:
        """Process text with financial-aware normalization"""
        text = text.lower().strip()
        text = self._normalize_financial_terms(text)
        text = self._lemmatize_text(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize text while preserving financial terms"""
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def _normalize_financial_terms(self, text: str) -> str:
        """Normalize financial quantities and metrics"""
        text = re.sub(self.FINANCIAL_PATTERNS['currency_amount'], 'CURRENCY_AMOUNT', text)
        text = re.sub(self.FINANCIAL_PATTERNS['financial_quantity'], 'FINANCIAL_QUANTITY', text)
        text = re.sub(self.FINANCIAL_PATTERNS['percentage'], 'PERCENTAGE_VALUE', text)
        return text