import time
from typing import Dict, List, Optional, Union, Any

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import numpy as np
import pandas as pd
import scipy

from III_feature_engineering_base import BaseFeatureExtractor, TextPreprocessor

from typing import Dict as DictType
from typing import List as ListType


class TextVectorizer(BaseFeatureExtractor):
    """Enhanced text vectorization with multiple methods"""
    
    VALID_METHODS = {'tfidf', 'bow', 'binary'}
    def __init__(self,
                 method: str = 'tfidf',
                 max_features: int = 5000,
                 ngram_range: tuple = (1, 2),
                 min_df: Union[int, float] = 2,
                 max_df: Union[int, float] = 0.95,
                 vocabulary: Optional[dict] = None,
                 stop_words: Optional[Union[str, list]] = 'english',
                 token_pattern: str = r'(?u)\b\w+\b'):

        """
        Args:
            method: Vectorization method ('tfidf', 'bow', or 'binary')
            max_features: Maximum number of features
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            vocabulary: Optional pre-defined vocabulary
            stop_words: Stop words to remove
            token_pattern: Regular expression pattern for tokenization
        """
        super().__init__()
        
        if method not in self.VALID_METHODS:
            raise ValueError(f"Method must be one of {self.VALID_METHODS}")
        
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = vocabulary
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TextVectorizer':
        """Fit the vectorizer to the data"""
        self._validate_input(X, ['snippets'])
        
        try:
            texts = self._preprocess_texts(X)
            
            if self.method == 'tfidf':
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=self.max_df,
                    vocabulary=self.vocabulary,
                    stop_words=self.stop_words,
                    token_pattern=self.token_pattern
                )
            elif self.method == 'binary':
                self.vectorizer = CountVectorizer(
                    binary=True,
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=self.max_df,
                    vocabulary=self.vocabulary,
                    stop_words=self.stop_words,
                    token_pattern=self.token_pattern
                )
            else:  # bow
                self.vectorizer = CountVectorizer(
                    binary=False,
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=self.max_df,
                    vocabulary=self.vocabulary,
                    stop_words=self.stop_words,
                    token_pattern=self.token_pattern
                )
            
            self.vectorizer.fit(texts)
            self._is_fitted = True
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting vectorizer: {str(e)}")
            raise

    def transform(self, X: pd.DataFrame) -> scipy.sparse.spmatrix:
        """Transform the texts into document-term matrix"""
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        self._validate_input(X, ['snippets'])
        
        try:
            start_time = time.time()
            texts = self._preprocess_texts(X)
            features = self.vectorizer.transform(texts)
            
            # Calculate statistics
            self.stats = self._calculate_stats(features)
            self.stats.extraction_time = time.time() - start_time
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error transforming texts: {str(e)}")
            raise

    def _preprocess_texts(self, X: pd.DataFrame) -> List[str]:
        """Preprocess texts for vectorization"""
        return X['snippets'].apply(
            lambda snippets: self.preprocessor.clean_text(' '.join(snippets))
        ).tolist()

    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary terms)"""
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        return self.vectorizer.get_feature_names()

    def get_top_features(self, n: int = 10) -> Dict[str, float]:
        """Get top n features by their IDF scores (for TF-IDF) or frequency (for BOW)"""
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before getting top features")
            
        feature_names = self.get_feature_names()
        
        if self.method == 'tfidf':
            scores = self.vectorizer.idf_
        else:
            scores = np.asarray(self.vectorizer.vocabulary_.values())
            
        top_indices = np.argsort(scores)[-n:]
        return {
            feature_names[i]: float(scores[i])
            for i in top_indices
        }