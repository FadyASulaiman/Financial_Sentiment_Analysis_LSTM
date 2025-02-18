import time
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from III_feature_engineering_base import BaseFeatureExtractor


class TextVectorizer(BaseFeatureExtractor):
    """Text vectorization with TF-IDF or Bag-of-Words."""

    VALID_METHODS = {"tfidf", "bow", "binary"}

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 2000,
        ngram_range: tuple = (1, 2),
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.95,
        vocabulary: Optional[List[str]] = None,
        stop_words: Optional[Union[str, List[str]]] = "english",
    ):
        super().__init__()

        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method: {method}. Choose from {self.VALID_METHODS}")

        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = vocabulary
        self.stop_words = stop_words
        self.vectorizer: Optional[Union[TfidfVectorizer, CountVectorizer]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TextVectorizer":
        """Fit the vectorizer to the training data."""
        self._validate_input(X, ["clean_sentence"])

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                vocabulary=self.vocabulary,
                stop_words=self.stop_words,
            )
        elif self.method == 'binary':
                self.vectorizer = CountVectorizer(
                    binary=True,
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=self.max_df,
                    vocabulary=self.vocabulary,
                    stop_words=self.stop_words
                )
        else:  # bow
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                vocabulary=self.vocabulary,
                stop_words=self.stop_words,
            )

        self.vectorizer.fit(X["clean_sentence"])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> scipy.sparse.csr_matrix:
        """Transform text data into a document-term matrix."""
        self._validate_input(X, ["clean_sentence"])
        if not self.vectorizer:
            raise ValueError("Vectorizer must be fitted before transform")

        start_time = time.time()
        features = self.vectorizer.transform(X["clean_sentence"])
        self.stats = self._calculate_stats(features)
        self.stats.extraction_time = time.time() - start_time
        return features

    def get_feature_names_out(self) -> np.ndarray:
        """Get feature names (vocabulary terms)."""
        if not self.vectorizer:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        return self.vectorizer.get_feature_names_out()


    def get_top_features(self, n: int = 10) -> Dict[str, float]:
        """Get top n features by TF-IDF score or frequency."""
        if not self.vectorizer:
            raise ValueError("Vectorizer must be fitted before getting top features")

        feature_names = self.get_feature_names_out()

        if self.method == "tfidf":
            scores = self.vectorizer.idf_
        else:  # Bag-of-Words
            vocabulary = self.vectorizer.vocabulary_
            scores = np.array([vocabulary.get(fname, 0) for fname in feature_names])

        top_indices = np.argsort(scores)[-n:]
        return dict(zip(feature_names[top_indices], scores[top_indices]))