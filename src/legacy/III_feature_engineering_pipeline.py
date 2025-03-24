import logging
from typing import Union

import scipy
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from III_feature_engineering_base import TextPreprocessor
from III_feature_engineering_embedding_vectorizer import EmbeddingVectorizer
from III_feature_engineering_feature_extractor import BusinessFeatureExtractor, LinguisticFeatureExtractor
from III_feature_engineering_txt_vectorizer import TextVectorizer
from II_data_preprocessing import DataPreprocessor
from I_improved_data_loading_and_eda import FiqaDataLoading


# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FeatureCombiner:
    """Intelligent feature combiner with automatic type handling"""
    def __init__(self, max_components: int = 100):
        self.max_components = max_components
        self.reducer = None
        self.actual_components_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Calculate safe component count
        self.actual_components_ = self._calculate_safe_components(X)
        
        # Only apply PCA if beneficial
        if self.actual_components_ < X.shape[1] * 0.8:  # Only reduce if significant
            self.reducer = PCA(n_components=self.actual_components_)
            return self.reducer.fit_transform(X)
        return X

    def _calculate_safe_components(self, X: np.ndarray) -> int:
        """Dynamically determine maximum safe components"""
        max_possible = min(X.shape[0], X.shape[1])
        return min(self.max_components, max_possible)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reducer:
            return self.reducer.transform(X)
        return X

class FeatureEngineeringPipeline:
    """Optimized pipeline with financial-aware processing"""

    def __init__(self, embedding_path: str = None, use_gpu: bool = False):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TextVectorizer(financial_terms=["currency_amount", "financial_quantity"])
        self.linguistic = LinguisticFeatureExtractor()
        self.business = BusinessFeatureExtractor()
        self.embeddings = EmbeddingVectorizer(embedding_path) if embedding_path else None
        self.combiner = FeatureCombiner()
        self.use_gpu = use_gpu

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        # Handle text preprocessing
        if "clean_sentence" not in X.columns:
            X["clean_sentence"] = X["sentence"].apply(self.preprocessor.clean_text)
        
        # Feature extraction with dimension validation
        features = []
        for extractor in [self.vectorizer, self.linguistic, self.business]:
            feat = extractor.fit_transform(X)
            features.append(self._ensure_2d(feat))
            
        if self.embeddings:
            features.append(self._ensure_2d(self.embeddings.fit_transform(X)))
            
        combined = np.hstack(features)
        return self.combiner.fit_transform(combined)


    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Handle text preprocessing
        if "clean_sentence" not in X.columns:
            X["clean_sentence"] = X["sentence"].apply(self.preprocessor.clean_text)
        
        # Feature extraction with dimensional checks
        features = []
        for extractor in [self.vectorizer, self.linguistic, self.business]:
            feat = extractor.transform(X)
            features.append(self._ensure_2d(feat))
            
        if self.embeddings:
            features.append(self._ensure_2d(self.embeddings.transform(X)))
            
        # Validate dimensions before stacking
        self._validate_stack_dimensions(features)
            
        combined = np.hstack(features)
        return self.combiner.transform(combined)

    def _validate_stack_dimensions(self, features: list):
        """Ensure all features have compatible dimensions"""
        dims = [arr.ndim for arr in features]
        if len(set(dims)) > 1:
            raise ValueError(f"Inconsistent feature dimensions: {dims}")
            
        shapes = [arr.shape[0] for arr in features]
        if len(set(shapes)) > 1:
            raise ValueError(f"Mismatched sample counts: {shapes}")

    def _ensure_2d(self, array: Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
        """Consistent dimensional enforcement"""
        if isinstance(array, scipy.sparse.spmatrix):
            array = array.toarray()
        if array.ndim == 1:
            return array.reshape(-1, 1)
        if array.ndim == 3:  # Handle potential 3D outputs
            return array.reshape(array.shape[0], -1)
        return array


if __name__ == "__main__":

    loader = FiqaDataLoading("data/raw/FiQA_ABSA_task1/task1_headline_ABSA_train.json")
    df = loader.load_and_preprocess()

    preprocessor = DataPreprocessor(
        text_clean=True,
        lemmatize=True,
        remove_stopwords=True,
        custom_stopwords=['custom', 'words'],
        balance_strategy='hybrid',
        target_ratio=0.8,
        discrepancy_threshold=0.5,
        random_state=42
    )
    # Apply preprocessing
    cleaned_data = preprocessor.transform(df)

    print(f"Shape of preprocessed DF: {cleaned_data.shape}")


    # pipeline_config = {
    #     "embedding_path": None,  # Replace with your embedding file path or None
    #     "embedding_format": "word2vec",  # Or 'glove', 'fasttext' if applicable
    #     "embedding_aggregation": "mean", # Or 'max', 'sum'
    #     "transformer_model_name": "bert-base-uncased",
    #     "transformer_pooling": "mean",  # Or 'cls', 'max'
    #     "max_features_tfidf": 5000,
    #     "ngram_range_tfidf": (1, 2),
    #     "batch_size": 32,  # Adjust based on your resources
    #     "cache_dir": './cache',  # Optional cache directory
    #     "combiner_config": {
    #         "scaler": "standard",  # Optional scaling
    #         "reducer": "pca",      # Optional dimensionality reduction
    #         "n_components": 0.95,   # Number of components or variance ratio to keep
    #     }
    # }

    # pipeline = FeatureEngineeringPipeline(**pipeline_config)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_data, cleaned_data['sentiment_score'], test_size=0.2, random_state=42
    )

    pipeline = FeatureEngineeringPipeline("data/embeddings")

    # Fit and transform
    X_train_transformed = pipeline.fit_transform(X_train)

    # Transform the test data
    X_test_transformed = pipeline.transform(X_test)

    # Now you can use X_train_transformed and X_test_transformed with your machine learning model
    print("X_train transformed shape:", X_train_transformed.shape)
    print("X_test transformed shape:", X_test_transformed.shape)
