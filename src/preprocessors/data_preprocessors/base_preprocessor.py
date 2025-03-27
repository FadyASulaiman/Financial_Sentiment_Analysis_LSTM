from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessorBase(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for text preprocessors"""
    
    @abstractmethod
    def fit(self, X, y=None):
        """Fit method implementation required by scikit-learn"""
        pass
    
    @abstractmethod
    def transform(self, X):
        """Transform method implementation required by scikit-learn"""
        pass
    
    def fit_transform(self, X, y=None):
        """Fit and transform data"""
        return self.fit(X, y).transform(X)