from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessorBase(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for text preprocessors"""

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)