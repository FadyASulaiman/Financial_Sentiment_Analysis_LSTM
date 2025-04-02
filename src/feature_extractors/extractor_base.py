from abc import ABC, abstractmethod
from typing import Dict, List
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractorBase(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for all feature extractors"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def fit(self, X, y=None):

        return self
    
    @abstractmethod
    def transform(self, X):
        pass
    
    def get_feature_names(self) -> List[str]:
        return []