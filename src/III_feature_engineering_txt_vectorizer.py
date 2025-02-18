import time
from typing import List, Optional, Union

import logging 


import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from III_feature_engineering_base import BaseFeatureExtractor


class TextVectorizer(BaseFeatureExtractor):
    """Enhanced text vectorization with improved feature handling"""

    VALID_METHODS = {"tfidf", "bow", "binary"}

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.95,
        stop_words: Optional[Union[str, List[str]]] = "english",
        financial_terms: Optional[List[str]] = None
    ):
        super().__init__()
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.financial_terms = financial_terms or []
        self.vectorizer = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TextVectorizer":
        self._validate_input(X, ["clean_sentence"])
        
        # Create custom vocabulary for financial terms
        custom_vocab = set(self.financial_terms)
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=self.stop_words,
                vocabulary=list(custom_vocab) if custom_vocab else None
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=self.stop_words,
                binary=(self.method == "binary"),
                vocabulary=list(custom_vocab) if custom_vocab else None
            )
            
        self.vectorizer.fit(X["clean_sentence"])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> scipy.sparse.csr_matrix:
        self._validate_input(X, ["clean_sentence"])
        return self.vectorizer.transform(X["clean_sentence"])

    def get_feature_names_out(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]:
        """Add statistics tracking to existing implementation"""
        start_time = time.time()
        features = super().fit_transform(X, y)
        self.stats.extraction_time = time.time() - start_time 
        return features
