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
    
    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.scaler = None
        self.reducer = None
        self.feature_types_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Dynamically detect feature types
        self.feature_types_ = self._detect_feature_types(X)
        
        # Build appropriate pipeline
        if self.feature_types_ == 'text_only':
            processor = PCA(n_components=self.n_components)
        elif self.feature_types_ == 'non_text_only':
            processor = Pipeline([
                ('scaler', StandardScaler()),
                ('reducer', PCA(n_components=self.n_components))
            ])
        else:
            processor = Pipeline([
                ('col_trans', ColumnTransformer([
                    ('text', 'passthrough', slice(0, self.text_boundary_)),
                    ('other', StandardScaler(), slice(self.text_boundary_, None))
                ])),
                ('reducer', PCA(n_components=self.n_components))
            ])
            
        return processor.fit_transform(X)

    def _detect_feature_types(self, X: np.ndarray) -> str:
        """Identify feature composition types"""
        # Implement your actual feature type detection logic here
        # This example assumes first 5000 features are text-based
        self.text_boundary_ = min(5000, X.shape[1])
        
        if X.shape[1] == 0:
            raise ValueError("Received empty feature matrix")
        elif self.text_boundary_ == X.shape[1]:
            return 'text_only'
        elif self.text_boundary_ == 0:
            return 'non_text_only'
        return 'mixed'

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Implement corresponding transform logic
        pass

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
        if "clean_sentence" not in X.columns:
            X["clean_sentence"] = X["sentence"].apply(self.preprocessor.clean_text)
            
        features = [
            self.vectorizer.transform(X),
            self.linguistic.transform(X),
            self.business.transform(X)
        ]
        
        if self.embeddings:
            features.append(self.embeddings.transform(X))
            
        combined = np.hstack(features)
        return self.combiner.transform(combined)
    
    def _ensure_2d(self, array: Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
        """Ensure all feature arrays are 2-dimensional"""
        if isinstance(array, scipy.sparse.spmatrix):
            array = array.toarray()
        if array.ndim == 1:
            return array.reshape(-1, 1)
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

    pipeline = FeatureEngineeringPipeline()

    # Fit and transform
    X_train_transformed = pipeline.fit_transform(X_train)


    # # Transform the test data
    X_test_transformed = pipeline.transform(X_test)

    # Now you can use X_train_transformed and X_test_transformed with your machine learning model
    print("X_train transformed shape:", X_train_transformed.shape)
    print("X_test transformed shape:", X_test_transformed.shape)

