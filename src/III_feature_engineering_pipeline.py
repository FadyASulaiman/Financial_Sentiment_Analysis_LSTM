import logging
from pathlib import Path
import time
import scipy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import warnings
from typing import Union, List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from tqdm import tqdm

from III_feature_engineering_base import BaseFeatureExtractor
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

class FeatureCombiner(BaseFeatureExtractor):
    """Enhanced feature combination and selection"""
    
    VALID_SCALERS = {'standard', 'minmax', 'robust', None}
    VALID_REDUCERS = {'pca', 'truncated_svd', 'selection', None}
    def __init__(self,
                 scaler: str = 'standard',
                 reducer: str = None,
                 n_components: Union[int, float] = 0.95,
                 feature_selection_method: str = 'mutual_info',
                 correlation_threshold: float = 0.95,
                 min_variance: float = 0.01,
                 imputation_strategy: str = 'mean',
                 random_state: int = 42):
        """
        Args:
            scaler: Type of scaling to apply
            reducer: Type of dimensionality reduction
            n_components: Number of components or variance ratio to keep
            feature_selection_method: Method for feature selection
            correlation_threshold: Threshold for correlation-based feature removal
            min_variance: Minimum variance threshold for feature removal
            imputation_strategy: Strategy for handling missing values ('mean', 'median', 'most_frequent', 'constant')
            random_state: Random state for reproducibility
        """
        super().__init__()
        
        if scaler not in self.VALID_SCALERS:
            raise ValueError(f"Scaler must be one of {self.VALID_SCALERS}")
        if reducer not in self.VALID_REDUCERS:
            raise ValueError(f"Reducer must be one of {self.VALID_REDUCERS}")
        
        self.scaler = scaler
        self.reducer = reducer
        self.n_components = n_components
        self.feature_selection_method = feature_selection_method
        self.correlation_threshold = correlation_threshold
        self.min_variance = min_variance
        self.imputation_strategy = imputation_strategy
        self.random_state = random_state
        
        # Initialize components
        self.imputer = None
        self.scaler_obj = None
        self.reducer_obj = None
        self.selected_features = None
        self.feature_importances_ = None
        self.correlation_matrix_ = None
        self.stats = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[pd.Series] = None) -> 'FeatureCombiner':
        """Fit the feature combiner with dimension validation"""
        try:
            start_time = time.time()
            
            # Convert to DataFrame if necessary
            X = self._ensure_dataframe(X)
            
            # Initialize and fit imputer
            self.imputer = SimpleImputer(strategy=self.imputation_strategy)
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
            
            # Remove low variance features
            X = self._remove_low_variance(X)
            if X.shape[1] == 0:
                raise ValueError("No features left after variance threshold")
            
            # Remove highly correlated features
            X = self._remove_correlations(X)
            if X.shape[1] == 0:
                raise ValueError("No features left after correlation removal")
            
            # Initialize and fit scaler
            if self.scaler:
                self._fit_scaler(X)
                X = pd.DataFrame(self.scaler_obj.transform(X), columns=X.columns)
            
            # Initialize and fit reducer
            if self.reducer:
                self._fit_reducer(X, y)
            
            # Store feature importances if applicable
            self._calculate_feature_importances(X, y)
            
            # Calculate and store statistics
            self._calculate_stats(X, time.time() - start_time)
            
            self._is_fitted = True
            return self
            
        except Exception as e:
            self.logger.error(f"Error in feature combiner fitting: {str(e)}")
            raise

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform the features"""
        if not self._is_fitted:
            raise ValueError("FeatureCombiner must be fitted before transform")
        
        try:
            X = self._ensure_dataframe(X)
            
            # Apply imputation
            X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
            
            # Apply feature selection
            if self.selected_features is not None:
                X = X[self.selected_features]
            
            # Apply scaling
            if self.scaler_obj is not None:
                X = pd.DataFrame(self.scaler_obj.transform(X), columns=X.columns)
            
            # Apply reduction
            if self.reducer_obj is not None:
                X = self.reducer_obj.transform(X)
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error transforming features: {str(e)}")
            raise
            
    def _ensure_dataframe(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Ensure input is a DataFrame"""
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return X

    def _remove_low_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with variance below threshold"""
        variances = X.var()
        self.selected_features = variances[variances >= self.min_variance].index
        return X[self.selected_features]

    def _remove_correlations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        self.correlation_matrix_ = X.corr()
        
        # Create correlation matrix mask
        upper = np.triu(np.ones(self.correlation_matrix_.shape), k=1).astype(bool)
        high_corr_pairs = [(i, j) for i, j in zip(*np.where(upper & (np.abs(self.correlation_matrix_) >= self.correlation_threshold)))]
        
        # Remove features with high correlation
        features_to_remove = set()
        for f1, f2 in high_corr_pairs:
            f1_name, f2_name = X.columns[f1], X.columns[f2]
            f1_importance = self._get_feature_importance(X[f1_name], X.columns)
            f2_importance = self._get_feature_importance(X[f2_name], X.columns)
            
            if f1_importance < f2_importance:
                features_to_remove.add(f1_name)
            else:
                features_to_remove.add(f2_name)
        
        self.selected_features = [col for col in X.columns if col not in features_to_remove]
        return X[self.selected_features]

    def _fit_scaler(self, X: pd.DataFrame):
        """Initialize and fit the scaler"""
        if self.scaler == 'standard':
            self.scaler_obj = StandardScaler()
        elif self.scaler == 'minmax':
            self.scaler_obj = MinMaxScaler()
        elif self.scaler == 'robust':
            self.scaler_obj = RobustScaler()
        
        if self.scaler_obj is not None:
            self.scaler_obj.fit(X)


    def _fit_reducer(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Initialize and fit the reducer with proper dimension validation"""
        if self.reducer is None:
            return
        try:
            # Validate and get appropriate number of components
            n_components = self._validate_n_components(X)
            
            if self.reducer == 'pca':
                # Choose appropriate SVD solver based on data dimensions
                n_samples, n_features = X.shape
                if n_components == 'mle':
                    svd_solver = 'full'
                elif n_components < 1:
                    svd_solver = 'full'
                elif n_components < min(n_features, n_samples):
                    svd_solver = 'randomized'
                else:
                    svd_solver = 'full'

                self.reducer_obj = PCA(
                    n_components=n_components,
                    svd_solver=svd_solver,
                    random_state=self.random_state
                )
                
            elif self.reducer == 'truncated_svd':
                self.reducer_obj = TruncatedSVD(
                    n_components=n_components,
                    random_state=self.random_state
                )
                
            elif self.reducer == 'selection':
                self.reducer_obj = SelectKBest(
                    score_func=mutual_info_regression 
                    if self.feature_selection_method == 'mutual_info' 
                    else f_regression,
                    k=n_components
                )

            # Fit the reducer
            self.reducer_obj.fit(X, y)
            
            # Log explained variance if available
            if hasattr(self.reducer_obj, 'explained_variance_ratio_'):
                cumulative_variance = np.cumsum(self.reducer_obj.explained_variance_ratio_)
                self.logger.info(f"Cumulative explained variance ratio: {cumulative_variance[-1]:.4f}")

        except Exception as e:
            self.logger.error(f"Error in dimension reduction: {str(e)}")
            raise


    def _get_optimal_n_components(self, X: pd.DataFrame, variance_threshold: float = 0.95) -> int:
        """Determine optimal number of components using explained variance ratio"""
        try:
            # Fit a temporary PCA to analyze explained variance
            temp_pca = PCA(n_components=min(X.shape[0], X.shape[1]), random_state=self.random_state)
            temp_pca.fit(X)
            
            # Find number of components that explain desired variance
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            self.logger.info(f"Optimal n_components for {variance_threshold} variance: {n_components}")
            return n_components
            
        except Exception as e:
            self.logger.warning(f"Error in optimal n_components calculation: {str(e)}")
            return min(X.shape[0], X.shape[1])
        

    def _calculate_feature_importances(self, X: pd.DataFrame, y: Optional[pd.Series]):
        """Calculate feature importances"""
        if y is not None and self.feature_selection_method == 'mutual_info':
            importances = mutual_info_regression(X, y)
        else:
            importances = np.var(X, axis=0)
        
        self.feature_importances_ = dict(zip(X.columns, importances))

    def _get_feature_importance(self, feature: pd.Series, all_features: pd.Index) -> float:
        """Get importance score for a single feature"""
        if self.feature_importances_ is not None:
            return self.feature_importances_.get(feature.name, 0)
        return feature.var()

    def _calculate_stats(self, X: pd.DataFrame, processing_time: float):
        """Calculate and store combination statistics with dimension information"""
        n_samples, n_features = X.shape
        self.stats = {
            'original_shape': X.shape,
            'n_components_possible': min(n_samples, n_features),
            'n_components_used': getattr(self.reducer_obj, 'n_components_', n_features),
            'explained_variance_ratio': getattr(self.reducer_obj, 'explained_variance_ratio_', None),
            'n_features_removed': {
                'variance': self._n_features_removed_variance,
                'correlation': self._n_features_removed_correlation
            },
            'imputation_stats': {
                'strategy': self.imputation_strategy,
                'n_imputed_values': np.sum(self.imputer.statistics_),
                'imputed_values_ratio': np.sum(self.imputer.statistics_) / (n_samples * n_features)
            },
            'processing_time': processing_time
        }
class FeatureEngineeringPipeline:
    """Orchestrates the complete feature engineering process"""



    def __init__(self,
                 config: Dict[str, Dict],
                 n_jobs: int = -1,
                 verbose: bool = True,
                 cache_dir: Optional[str] = None,
                 handle_errors: str = 'raise'):
        """
        Args:
            config: Configuration dictionary for feature extractors
            n_jobs: Number of jobs for parallel processing
            verbose: Whether to show progress information
            cache_dir: Directory for caching intermediate results
            handle_errors: How to handle errors ('raise' or 'ignore')
        """
        self.config = config
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.handle_errors = handle_errors
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.extractors = {}
        self.combiner = None
        self.stats = {}
        
        self._initialize_pipeline()



    def _initialize_pipeline(self):
        """Initialize all components of the pipeline"""
        try:
            # Initialize text vectorizer
            if 'text' in self.config:
                self.extractors['text'] = TextVectorizer(**self.config['text'])
            
            # Initialize embedding vectorizer
            if 'embeddings' in self.config:
                self.extractors['embeddings'] = EmbeddingVectorizer(**self.config['embeddings'])
            
            # Initialize transformer embeddings
            if 'transformers' in self.config:
                self.extractors['transformers'] = TransformerEmbeddings(**self.config['transformers'])
            
            # Initialize linguistic features
            if 'linguistic' in self.config:
                self.extractors['linguistic'] = LinguisticFeatureExtractor(**self.config['linguistic'])
            
            # Initialize business features
            if 'business' in self.config:
                self.extractors['business'] = BusinessFeatureExtractor(**self.config['business'])
            
            # Initialize feature combiner
            if 'combiner' in self.config:
                self.combiner = FeatureCombiner(**self.config['combiner'])
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {str(e)}")
            raise


    def _get_cache_path(self, name: str, X: pd.DataFrame) -> Optional[Path]:
        """Get cache path for feature matrix with safe hashing"""
        if not self.cache_dir:
            return None
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create a hashable representation of the data
            hash_data = X.copy()
            
            # Convert list columns to strings
            for col in hash_data.columns:
                if hash_data[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    hash_data[col] = hash_data[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
            
            # Create cache key based on data characteristics
            data_hash = pd.util.hash_pandas_object(hash_data).sum()
            
            # Include feature extractor configuration in the hash
            config_str = str(self.config.get(name, {}))
            combined_hash = hash(f"{data_hash}_{config_str}")
            
            return self.cache_dir / f"{name}_features_{abs(combined_hash)}.npz"
            
        except Exception as e:
            self.logger.warning(f"Cache path generation failed: {str(e)}. Proceeding without caching.")
            return None

    def _save_to_cache(self, features: Union[np.ndarray, scipy.sparse.spmatrix], path: Path) -> bool:
        """Save features to cache with error handling"""
        try:
            if path.suffix == '.npz':
                if scipy.sparse.issparse(features):
                    scipy.sparse.save_npz(str(path), features)
                else:
                    np.savez_compressed(str(path), features=features)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {str(e)}")
            return False

    def _load_from_cache(self, path: Path) -> Optional[Union[np.ndarray, scipy.sparse.spmatrix]]:
        """Load features from cache with error handling"""
        try:
            if not path.exists():
                return None
                
            if path.suffix == '.npz':
                try:
                    # Try loading as sparse matrix first
                    return scipy.sparse.load_npz(str(path))
                except Exception:
                    # If that fails, try loading as dense array
                    with np.load(str(path)) as data:
                        return data['features']
            return None
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {str(e)}")
            return None

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform with improved error handling and caching"""
        try:
            start_time = time.time()
            features = []
            
            # Process each feature extractor
            for name, extractor in tqdm(self.extractors.items(), desc="Processing feature extractors"):
                if self.verbose:
                    self.logger.info(f"Processing {name} features...")
                
                try:
                    # Check cache
                    cache_path = self._get_cache_path(name, X)
                    feature_matrix = None
                    
                    if cache_path:
                        feature_matrix = self._load_from_cache(cache_path)
                    
                    if feature_matrix is None:
                        # Transform data
                        feature_matrix = extractor.fit_transform(X)
                        
                        # Save to cache if possible
                        if cache_path:
                            self._save_to_cache(feature_matrix, cache_path)
                    
                    features.append(feature_matrix)
                    self.stats[name] = extractor.stats
                    
                except Exception as e:
                    self.logger.error(f"Error processing {name} features: {str(e)}")
                    if self.handle_errors == 'ignore':
                        continue
                    raise
            
            # Combine features
            if self.verbose:
                self.logger.info("Combining features...")
            
            if not features:
                raise ValueError("No features were successfully extracted")
            
            # Convert all features to dense if needed
            processed_features = []
            for f in features:
                if scipy.sparse.issparse(f):
                    processed_features.append(f.toarray())
                else:
                    processed_features.append(f)
            
            combined_features = np.hstack(processed_features)
            
            # Apply feature combination if configured
            if self.combiner is not None:
                combined_features = self.combiner.fit_transform(combined_features, y)
                self.stats['combiner'] = self.combiner.stats
            
            self.stats['total_time'] = time.time() - start_time
            
            if self.verbose:
                self._log_pipeline_summary()
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise


    def _log_pipeline_summary(self):
        """Log summary of the feature engineering process"""
        self.logger.info("\n=== Feature Engineering Summary ===")
        
        for name, stats in self.stats.items():
            if name == 'total_time':
                continue
            
            self.logger.info(f"\n{name.upper()} Features:")
            self.logger.info(f"- Number of features: {stats.n_features}")
            self.logger.info(f"- Memory usage: {stats.memory_usage:.2f} MB")
            self.logger.info(f"- Processing time: {stats.extraction_time:.2f}s")
            
            if hasattr(stats, 'sparsity'):
                self.logger.info(f"- Sparsity: {stats.sparsity:.2%}")
            
            if name == 'combiner':
                self.logger.info(f"- Reduction ratio: {stats.reduction_ratio:.2%}")
                if stats.explained_variance_ratio is not None:
                    self.logger.info(f"- Explained variance ratio: {sum(stats.explained_variance_ratio):.2%}")
        
        self.logger.info(f"\nTotal processing time: {self.stats['total_time']:.2f}s")

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        feature_names = []
        for name, extractor in self.extractors.items():
            if hasattr(extractor, 'feature_names'):
                prefix = f"{name}_"
                feature_names.extend([f"{prefix}{f}" for f in extractor.feature_names])
        return feature_names
    




if __name__ == "__main__":
    # Configuration
    config = {
        'text': {
            'method': 'tfidf',
            'max_features': 5000
        },
        'embeddings': {
            'embedding_path': 'path/to/embeddings',
            'embedding_dim': 300
        },
        'transformers': {
            'model_name': 'bert-base-uncased',
            'batch_size': 32
        },
        'linguistic': {
            'feature_groups': {'pos': True, 'syntax': True, 'sentiment': True}
        },
        'business': {
            'custom_entities': {'companies': ['Apple', 'Google', 'Microsoft']}
        },
        'combiner': {
            'scaler': 'standard',
            'reducer': 'pca',
            'n_components': 0.95,
            'imputation_strategy':'mean',
            'random_state':42
        }
    }


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

    # Initialize and run pipeline
    pipeline = FeatureEngineeringPipeline(
        config=config,
        n_jobs=-1,
        verbose=True,
        cache_dir='./cache',
        handle_errors='ignore'  # Continue even if some extractors fail
    )

    # Fit and transform
    features = pipeline.fit_transform(cleaned_data)

    # Get feature names
    feature_names = pipeline.get_feature_names()