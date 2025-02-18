import logging
from pathlib import Path
import time
import scipy

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler, StandardScaler

from typing import Union, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from III_feature_engineering_base import TextPreprocessor
from III_feature_engineering_embedding_vectorizer import EmbeddingVectorizer, TransformerEmbeddings
from III_feature_engineering_feature_extractor import BusinessFeatureExtractor, LinguisticFeatureExtractor
from III_feature_engineering_txt_vectorizer import TextVectorizer
from II_data_preprocessing import DataPreprocessor
from I_improved_data_loading_and_eda import FiqaDataLoading

@dataclass
class FeatureCombinerStats:
    """Statistics for feature combination process"""
    original_features: int
    final_features: int
    reduction_ratio: float
    feature_importances: Dict[str, float]
    correlation_matrix: Optional[pd.DataFrame]
    explained_variance_ratio: Optional[float]
    memory_usage: float
    processing_time: float

class FeatureCombiner:
    """Combines, scales, and reduces the dimensionality of features."""

    def __init__(
        self,
        scaler: Optional[str] = "standard",
        reducer: Optional[str] = None,
        n_components: Union[int, float] = 0.95,
        random_state: int = 42,
    ):
        self.scaler = scaler
        self.reducer = reducer
        self.n_components = n_components
        self.random_state = random_state
        self.scaler_obj = None
        self.reducer_obj = None
        self.stats: Dict = {}

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform the data."""
        start_time = time.time()
        X = self._scale(X, fit=True)
        X = self._reduce(X, fit=True, y=y)
        self.stats["processing_time"] = time.time() - start_time
        self.stats["n_samples"] = X.shape[0]
        self.stats["n_features"] = X.shape[1]
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        X = self._scale(X, fit=False)
        X = self._reduce(X, fit=False)
        return X

    def _scale(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Scale the data."""
        if self.scaler:
            if fit:
                if self.scaler == "standard":
                    self.scaler_obj = StandardScaler()
                elif self.scaler == "robust":
                    self.scaler_obj = RobustScaler()
                else:
                    raise ValueError(f"Invalid scaler: {self.scaler}")
                return self.scaler_obj.fit_transform(X)
            else:
                if not self.scaler_obj:
                    raise ValueError("Scaler must be fitted before transform")
                return self.scaler_obj.transform(X)
        return X

    def _reduce(self, X: np.ndarray, fit: bool = True, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Reduce the dimensionality of the data."""
        if self.reducer:
            if fit:
                if self.reducer == "pca":
                    from sklearn.decomposition import PCA

                    self.reducer_obj = PCA(
                        n_components=self.n_components, random_state=self.random_state
                    )
                elif self.reducer == "svd":
                    from sklearn.decomposition import TruncatedSVD

                    self.reducer_obj = TruncatedSVD(
                        n_components=self.n_components, random_state=self.random_state
                    )
                else:
                    raise ValueError(f"Invalid reducer: {self.reducer}")
                return self.reducer_obj.fit_transform(X, y=y)
            else:
                if not self.reducer_obj:
                    raise ValueError("Reducer must be fitted before transform")
                return self.reducer_obj.transform(X)
        return X


class FeatureEngineeringPipeline:
    """Orchestrates the feature engineering pipeline."""

    def __init__(
        self,
        embedding_path: Optional[Union[str, Path]] = None,
        embedding_format: str = "word2vec",
        embedding_aggregation: str = "mean",
        transformer_model_name: str = "bert-base-uncased",
        transformer_pooling: str = "mean",
        max_features_tfidf: int = 5000,
        ngram_range_tfidf: Tuple[int, int] = (1, 2),
        batch_size: int = 256,
        cache_dir: Optional[str] = None,
        combiner_config: Optional[Dict] = None,
    ):
        self.text_preprocessor = TextPreprocessor()
        self.linguistic_extractor = LinguisticFeatureExtractor(batch_size=batch_size)
        self.business_extractor = BusinessFeatureExtractor(batch_size=batch_size)
        self.tfidf_vectorizer = TextVectorizer(
            method="tfidf",
            max_features=max_features_tfidf,
            ngram_range=ngram_range_tfidf,
        )
        self.embedding_vectorizer = None
        if embedding_path:
            self.embedding_vectorizer = EmbeddingVectorizer(
                embedding_path=embedding_path,
                embedding_format=embedding_format,
                aggregation=embedding_aggregation,
                cache_dir=cache_dir,
            )

        self.transformer_embeddings = TransformerEmbeddings(
            model_name=transformer_model_name,
            pooling=transformer_pooling,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
        self.combiner = None
        if combiner_config:
            self.combiner = FeatureCombiner(**combiner_config)
        self.logger = logging.getLogger(__name__)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fits and transforms the data through the entire pipeline."""
        start_time = time.time()
        X["clean_sentence"] = X["sentence"].apply(self.text_preprocessor.clean_text)

        feature_extractors = [
            ("linguistic", self.linguistic_extractor),
            ("business", self.business_extractor),
            ("tfidf", self.tfidf_vectorizer),
            ("transformer", self.transformer_embeddings),
        ]
        if self.embedding_vectorizer:
            feature_extractors.append(("embeddings", self.embedding_vectorizer))

        combined_features = FeatureUnion(feature_extractors, n_jobs=1).fit_transform(X, y=y) # Added n_jobs=1 to avoid issues with spaCy
        if scipy.sparse.issparse(combined_features):
            combined_features = combined_features.toarray()

        if self.combiner:
            combined_features = self.combiner.fit_transform(combined_features, y=y)

        end_time = time.time()
        self.logger.info(f"Feature engineering completed in {end_time - start_time:.2f} seconds")
        return combined_features

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms the data through the entire pipeline."""
        start_time = time.time()
        X["clean_sentence"] = X["sentence"].apply(self.text_preprocessor.clean_text)

        feature_extractors = [
            ("linguistic", self.linguistic_extractor),
            ("business", self.business_extractor),
            ("tfidf", self.tfidf_vectorizer),
            ("transformer", self.transformer_embeddings),
        ]
        if self.embedding_vectorizer:
            feature_extractors.append(("embeddings", self.embedding_vectorizer))

        combined_features = FeatureUnion(feature_extractors).transform(X)
        if scipy.sparse.issparse(combined_features):
            combined_features = combined_features.toarray()

        if self.combiner:
            combined_features = self.combiner.transform(combined_features)

        end_time = time.time()
        self.logger.info(f"Feature engineering completed in {end_time - start_time:.2f} seconds")
        return combined_features




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


    pipeline_config = {
        "embedding_path": None,  # Replace with your embedding file path or None
        "embedding_format": "word2vec",  # Or 'glove', 'fasttext' if applicable
        "embedding_aggregation": "mean", # Or 'max', 'sum'
        "transformer_model_name": "bert-base-uncased",
        "transformer_pooling": "mean",  # Or 'cls', 'max'
        "max_features_tfidf": 5000,
        "ngram_range_tfidf": (1, 2),
        "batch_size": 32,  # Adjust based on your resources
        "cache_dir": './cache',  # Optional cache directory
        "combiner_config": {
            "scaler": "standard",  # Optional scaling
            "reducer": "pca",      # Optional dimensionality reduction
            "n_components": 0.95,   # Number of components or variance ratio to keep
        }
    }

    pipeline = FeatureEngineeringPipeline(**pipeline_config)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_data, cleaned_data['sentiment_score'], test_size=0.2, random_state=42
    )


    # Fit and transform the training data
    X_train_transformed = pipeline.fit_transform(X_train, y=y_train)  # Pass y_train for fitting FeatureCombiner

    # Transform the test data
    X_test_transformed = pipeline.transform(X_test)

    # Now you can use X_train_transformed and X_test_transformed with your machine learning model
    print("X_train transformed shape:", X_train_transformed.shape)
    print("X_test transformed shape:", X_test_transformed.shape)



    # # Delete
    # print("Any NaN values:", np.isnan(X).any())
    # print("NaN locations:", np.where(np.isnan(X)))

    # # 1. Check which features these columns correspond to
    # print("Shape of X:", X.shape)

    # # 2. If X was created from a DataFrame, check the original columns
    # if isinstance(X, np.ndarray):
    #     # Get the count of NaNs per column
    #     nan_counts = np.isnan(X).sum(axis=0)
    #     print("\nNumber of NaNs per column:")
    #     for i, count in enumerate(nan_counts):
    #         if count > 0:
    #             print(f"Column {i}: {count} NaNs")

    # # 3. Look at a sample of rows with NaNs
    # print("\nSample of rows with NaNs:")
    # nan_rows = np.where(np.isnan(X).any(axis=1))[0]
    # print(X[nan_rows[0]])

    # from sklearn.impute import SimpleImputer

    # # Create imputer
    # imputer = SimpleImputer(strategy='mean')  # for numerical features
    # X = imputer.fit_transform(X)

