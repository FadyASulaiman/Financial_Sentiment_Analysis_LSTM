import os
import re
import spacy
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from typing import List, Optional, Union, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from logging import Logger
import logging
from dataclasses import dataclass
from collections import Counter

from I_improved_data_loading_and_eda import FiqaDataLoading

@dataclass
class PreprocessingStats:
    """Statistics collected during preprocessing"""
    original_rows: int
    final_rows: int
    removed_rows: Dict[str, int]
    class_distribution: Counter
    missing_values: Dict[str, int]
    invalid_scores: int
    high_discrepancy_count: int



# =================================================
# ======= 1. Handle Nulls & Missing values ========
# =================================================

class MissingValueHandler:
    """Handles all missing value operations"""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.stats = {}

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame"""
        original_rows = len(df)
        missing_before = df.isnull().sum().to_dict()
        
        try:
            df = self._remove_critical_nulls(df)
            df = self._handle_aspect_nulls(df)
            
            self.stats = {
                'original_rows': original_rows,
                'final_rows': len(df),
                'removed_rows': original_rows - len(df),
                'missing_before': missing_before,
                'missing_after': df.isnull().sum().to_dict()
            }
            
            self._log_missing_value_stats()
            return df
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise

    def _remove_critical_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        critical_columns = ['sentence', 'sentiment_score', 'target']
        before_len = len(df)
        df = df.dropna(subset=critical_columns)
        dropped = before_len - len(df)
        if dropped > 0:
            self.logger.warning(f"Dropped {dropped} rows with missing critical data")
        return df

    def _handle_aspect_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        df['aspects'] = df['aspects'].fillna('').apply(
            lambda x: ['Unknown'] if not x else x)
        return df

    def _log_missing_value_stats(self):
        self.logger.info("=== Missing Value Handling Statistics ===")
        self.logger.info(f"Original rows: {self.stats['original_rows']}")
        self.logger.info(f"Final rows: {self.stats['final_rows']}")
        self.logger.info(f"Removed rows: {self.stats['removed_rows']}")



# =================================================
# ========= 2. Preprocess String Columns ==========
# =================================================

class TextColumnProcessor:
    """Processes text columns with advanced features"""
    
    def __init__(self, text_cleaner: 'TextCleaner', logger: Optional[Logger] = None):
        self.text_cleaner = text_cleaner
        self.logger = logger or logging.getLogger(__name__)
        self.stats = {'processed_sentences': 0, 'generated_snippets': 0}

    def process_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            
            # Clean the main sentence
            df['clean_sentence'] = df['sentence'].apply(self._clean_and_log)
            
            # Extract and validate snippets
            df['snippets'] = df['clean_sentence'].apply(self._extract_snippets)
            
            self._log_processing_stats()
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing text columns: {str(e)}")
            raise

    def _clean_and_log(self, text: str) -> str:
        cleaned = self.text_cleaner.clean_text(text)
        self.stats['processed_sentences'] += 1
        return cleaned

    def _extract_snippets(self, text: str) -> List[str]:
        # Enhanced snippet extraction with business-specific patterns
        splits = re.split(
            r'(?:\.|\;|\:|\!|\?|\n|\s+-\s+|(?<=\w)(?=[A-Z][a-z]))', 
            text
        )
        
        snippets = [
            snippet.strip() 
            for snippet in splits 
            if self._is_valid_snippet(snippet)
        ]
        
        if not snippets and text:
            snippets = [text]
            
        self.stats['generated_snippets'] += len(snippets)
        return snippets

    @staticmethod
    def _is_valid_snippet(snippet: str) -> bool:
        """Validate snippet quality"""
        snippet = snippet.strip()
        words = snippet.split()
        return (
            len(words) >= 2 and  # Minimum 2 words
            not all(len(word) <= 2 for word in words) and  # Not all short words
            any(len(word) > 3 for word in words)  # At least one substantial word
        )

    def _log_processing_stats(self):
        self.logger.info("=== Text Processing Statistics ===")
        self.logger.info(f"Processed sentences: {self.stats['processed_sentences']}")
        self.logger.info(f"Generated snippets: {self.stats['generated_snippets']}")



# =================================================
# ============= 3. Clean Text Columns =============
# =================================================

class TextCleaner:
    """Enhanced text cleaning operations"""
    
    def __init__(self, 
                 nlp_model,
                 lemmatize: bool = True, 
                 remove_stopwords: bool = True,
                 custom_stopwords: Optional[List[str]] = None,
                 logger: Optional[Logger] = None):
        self.nlp_model = nlp_model
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.custom_stopwords = set(custom_stopwords or [])
        self.logger = logger or logging.getLogger(__name__)
        
        # Compile regex patterns once
        self.special_chars_pattern = re.compile(r'[^\w\s.,!?-]')
        self.whitespace_pattern = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            self.logger.warning(f"Received non-string input: {type(text)}")
            text = str(text)

        try:
            text = self._remove_html(text)
            text = self._remove_special_chars(text)
            text = self._normalize_whitespace(text)
            return self._process_with_spacy(text.lower())
        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
            return text

    def _remove_html(self, text: str) -> str:
        return BeautifulSoup(text, "html.parser").get_text()

    def _remove_special_chars(self, text: str) -> str:
        text = text.replace('.', '')
        text = text.replace(',', '')
        text = text.replace(':', '')
        text = text.replace('&', '')
        text = text.replace('-', '')
        text = text.replace('$', '')

        return self.special_chars_pattern.sub('', text)

    def _normalize_whitespace(self, text: str) -> str:
        return self.whitespace_pattern.sub(" ", text).strip()

    def _process_with_spacy(self, text: str) -> str:
        doc = self.nlp_model(text)
        tokens = []
        
        for token in doc:
            if self._should_keep_token(token):
                processed_token = token.lemma_ if self.lemmatize else token.text
                tokens.append(processed_token.strip())

        return " ".join(tokens)

    def _should_keep_token(self, token) -> bool:
        return (
            token.text.strip() and
            not (self.remove_stopwords and 
                 (token.is_stop or token.text in self.custom_stopwords))
        )
    


# =================================================
# ========== 4. Validate Full DataFrame  ==========
# =================================================

class DataValidator:
    """Enhanced data validation operations with detailed reporting"""
    
    def __init__(self, 
                 discrepancy_threshold: float = 0.5,
                 logger: Optional[Logger] = None):
        self.sia = SentimentIntensityAnalyzer()
        self.discrepancy_threshold = discrepancy_threshold
        self.logger = logger or logging.getLogger(__name__)
        self.validation_stats = {}

    def validate_sentiment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate sentiment scores with comprehensive checks"""
        try:
            self._check_score_range(df)
            df = self._cross_validate_with_vader(df)
            self._log_validation_stats()
            return df
        except Exception as e:
            self.logger.error(f"Error in sentiment validation: {str(e)}")
            raise

    def _check_score_range(self, df: pd.DataFrame):
        invalid_mask = ~df['sentiment_score'].between(-1, 1)
        invalid_scores = df[invalid_mask]
        
        self.validation_stats['invalid_scores'] = {
            'count': len(invalid_scores),
            'details': invalid_scores[['sentiment_score']].to_dict() if len(invalid_scores) > 0 else {}
        }
        
        if not invalid_scores.empty:
            self.logger.warning(
                f"Found {len(invalid_scores)} invalid sentiment scores. "
                f"Range: [{df['sentiment_score'].min()}, {df['sentiment_score'].max()}]"
            )

    def _cross_validate_with_vader(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['vader_score'] = df['snippets'].apply(self._calculate_vader_score)
        df['score_discrepancy'] = (df['sentiment_score'] - df['vader_score']).abs()
        
        self._analyze_discrepancies(df)
        
        # Clean up temporary columns
        return df.drop(columns=['vader_score', 'score_discrepancy'])

    def _calculate_vader_score(self, snippets: List[str]) -> float:
        if not snippets:
            return 0.0
        
        # Calculate weighted average of snippet scores
        total_score = 0
        total_weight = 0
        
        for snippet in snippets:
            weight = len(snippet.split())  # Weight by number of words
            score = self.sia.polarity_scores(snippet)['compound']
            total_score += score * weight
            total_weight += weight
            
        return total_score / total_weight if total_weight > 0 else 0.0

    def _analyze_discrepancies(self, df: pd.DataFrame):
        high_discrepancy = df[df['score_discrepancy'] > self.discrepancy_threshold]
        
        self.validation_stats['discrepancies'] = {
            'total_analyzed': len(df),
            'high_discrepancy_count': len(high_discrepancy),
            'avg_discrepancy': df['score_discrepancy'].mean(),
            'max_discrepancy': df['score_discrepancy'].max(),
            'discrepancy_distribution': {
                'low': len(df[df['score_discrepancy'] <= 0.2]),
                'medium': len(df[(df['score_discrepancy'] > 0.2) & (df['score_discrepancy'] <= 0.5)]),
                'high': len(df[df['score_discrepancy'] > 0.5])
            }
        }

    def _log_validation_stats(self):
        self.logger.info("=== Sentiment Validation Statistics ===")
        self.logger.info(f"Invalid scores: {self.validation_stats['invalid_scores']['count']}")
        self.logger.info(f"High discrepancy cases: {self.validation_stats['discrepancies']['high_discrepancy_count']}")
        self.logger.info(f"Average discrepancy: {self.validation_stats['discrepancies']['avg_discrepancy']:.3f}")



# =================================================
# ========= 5. Balance Sentiment Classes  =========
# =================================================

class DataBalancer:
    """Enhanced class balancing with multiple strategies"""
    
    VALID_STRATEGIES = {'undersample', 'oversample', 'hybrid'}
    
    def __init__(self, 
                 strategy: str = 'hybrid',
                 target_ratio: Optional[float] = None,
                 random_state: int = 42,
                 logger: Optional[Logger] = None):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Strategy must be one of {self.VALID_STRATEGIES}")
        
        self.strategy = strategy
        self.target_ratio = target_ratio
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)
        self.balance_stats = {}

    def balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance classes using the specified strategy"""
        try:
            original_dist = df['sentiment_class'].value_counts()
            self.logger.info(f"Original class distribution:\n{original_dist}")
            
            if self.strategy == 'undersample':
                balanced_df = self._undersample(df)
            elif self.strategy == 'oversample':
                balanced_df = self._oversample(df)
            else:  # hybrid
                balanced_df = self._hybrid_sampling(df)
            
            self._calculate_balance_stats(original_dist, balanced_df)
            self._log_balance_stats()
            
            return balanced_df
            
        except Exception as e:
            self.logger.error(f"Error in class balancing: {str(e)}")
            raise

    def _undersample(self, df: pd.DataFrame) -> pd.DataFrame:
        class_counts = df['sentiment_class'].value_counts()
        min_class_size = class_counts.min()
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = df[df['sentiment_class'] == class_name]
            if len(class_df) > min_class_size:
                class_df = class_df.sample(n=min_class_size, random_state=self.random_state)
            balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True)

    def _oversample(self, df: pd.DataFrame) -> pd.DataFrame:
        class_counts = df['sentiment_class'].value_counts()
        max_class_size = class_counts.max()
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = df[df['sentiment_class'] == class_name]
            if len(class_df) < max_class_size:
                class_df = resample(
                    class_df,
                    replace=True,
                    n_samples=max_class_size,
                    random_state=self.random_state
                )
            balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True)

    def _hybrid_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        class_counts = df['sentiment_class'].value_counts()
        target_size = int(class_counts.median())
        
        if self.target_ratio:
            max_size = class_counts.max()
            target_size = int(max_size * self.target_ratio)
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = df[df['sentiment_class'] == class_name]
            if len(class_df) > target_size:
                class_df = class_df.sample(n=target_size, random_state=self.random_state)
            elif len(class_df) < target_size:
                class_df = resample(
                    class_df,
                    replace=True,
                    n_samples=target_size,
                    random_state=self.random_state
                )
            balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True)

    def _calculate_balance_stats(self, original_dist: pd.Series, balanced_df: pd.DataFrame):
        final_dist = balanced_df['sentiment_class'].value_counts()
        
        self.balance_stats = {
            'original_distribution': original_dist.to_dict(),
            'final_distribution': final_dist.to_dict(),
            'class_ratios': {
                'before': original_dist.max() / original_dist.min(),
                'after': final_dist.max() / final_dist.min()
            },
            'total_samples': {
                'before': len(original_dist),
                'after': len(balanced_df)
            }
        }

    def _log_balance_stats(self):
        self.logger.info("=== Class Balancing Statistics ===")
        self.logger.info(f"Strategy used: {self.strategy}")
        self.logger.info(f"Class ratio before: {self.balance_stats['class_ratios']['before']:.2f}")
        self.logger.info(f"Class ratio after: {self.balance_stats['class_ratios']['after']:.2f}")
        self.logger.info(f"Total samples before: {self.balance_stats['total_samples']['before']}")
        self.logger.info(f"Total samples after: {self.balance_stats['total_samples']['after']}")



# =================================================
# ============ Main Preprocessing Class ===========
# =================================================

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Orchestrator for the preprocessing pipeline"""
    
    def __init__(self,
                 text_clean: bool = True,
                 lemmatize: bool = True,
                 remove_stopwords: bool = True,
                 custom_stopwords: Optional[List[str]] = None,
                 balance_strategy: str = 'hybrid',
                 target_ratio: Optional[float] = None,
                 discrepancy_threshold: float = 0.5,
                 random_state: int = 42,
                 logger: Optional[Logger] = None):
        
        self.text_clean = text_clean
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.custom_stopwords = custom_stopwords
        self.balance_strategy = balance_strategy
        self.target_ratio = target_ratio
        self.discrepancy_threshold = discrepancy_threshold
        self.random_state = random_state
        self.logger = logger or self._setup_logger()
        
        # Initialize components
        self._initialize_components()
        self.preprocessing_stats = None

    def _setup_logger(self) -> Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _initialize_components(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.text_cleaner = TextCleaner(
            self.nlp,
            self.lemmatize,
            self.remove_stopwords,
            self.custom_stopwords,
            self.logger
        )
        self.text_processor = TextColumnProcessor(self.text_cleaner, self.logger)
        self.missing_handler = MissingValueHandler(self.logger)
        self.validator = DataValidator(self.discrepancy_threshold, self.logger)
        self.balancer = DataBalancer(
            self.balance_strategy,
            self.target_ratio,
            self.random_state,
            self.logger
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the preprocessing pipeline"""
        try:
            self.logger.info("Starting preprocessing pipeline...")
            df = X.copy()
            
            # Handle missing values
            df = self.missing_handler.handle_missing_values(df)
            
            # Clean text if requested
            if self.text_clean:
                df = self.text_processor.process_text_columns(df)
            
            # Validate sentiment scores
            df = self.validator.validate_sentiment_scores(df)
            
            # Balance classes if requested
            if self.balance_strategy:
                df = self.balancer.balance_classes(df)
            
            # Collect statistics
            self._collect_preprocessing_stats()
            
            self.logger.info("Preprocessing pipeline completed successfully")
            return df.reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

    def _collect_preprocessing_stats(self):
        """Collect statistics from all components"""
        self.preprocessing_stats = PreprocessingStats(
            original_rows=self.missing_handler.stats['original_rows'],
            final_rows=self.missing_handler.stats['final_rows'],
            removed_rows=self.missing_handler.stats['removed_rows'],
            class_distribution=self.balancer.balance_stats.get('final_distribution', {}),
            missing_values=self.missing_handler.stats['missing_after'],
            invalid_scores=self.validator.validation_stats['invalid_scores']['count'],
            high_discrepancy_count=self.validator.validation_stats['discrepancies']['high_discrepancy_count']
        )

    def get_preprocessing_summary(self) -> Dict:
        """Return a summary of the preprocessing operations"""
        if not self.preprocessing_stats:
            return {}
        
        return {
            'data_size': {
                'original': self.preprocessing_stats.original_rows,
                'final': self.preprocessing_stats.final_rows,
                'removed': self.preprocessing_stats.removed_rows
            },
            'quality_metrics': {
                'invalid_scores': self.preprocessing_stats.invalid_scores,
                'high_discrepancies': self.preprocessing_stats.high_discrepancy_count
            },
            'class_distribution': self.preprocessing_stats.class_distribution,
            'missing_values': self.preprocessing_stats.missing_values
        }



# =================================================
# ======= Save & Version Preprocessed Data ========
# =================================================

class PreprocessedDataVersioning:
    def save_versioned_dataframe(self, df: pd.DataFrame, base_path: str, base_filename: str):
            
        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Get list of existing versions
        existing_files = [f for f in os.listdir(base_path) 
                        if f.startswith(base_filename) and f.endswith('.csv')]
        
        # Determine next version number
        if not existing_files:
            version = 1
        else:
            versions = [int(f.split('_v')[1].split('.')[0]) for f in existing_files]
            version = max(versions) + 1
        
        filename = f"{base_filename}_v{version}.csv"
        full_path = os.path.join(base_path, filename)
        
        df.to_csv(full_path, index=False)
        print(f"DataFrame saved as: {filename}")
        
        return full_path
    

if __name__ == "__main__":
    # Initialize preprocessor with custom configuration
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

    loader = FiqaDataLoading("data/raw/FiQA_ABSA_task1/task1_headline_ABSA_train.json")
    df = loader.load_and_preprocess()

    # Apply preprocessing
    cleaned_data = preprocessor.transform(df)

    data_saver = PreprocessedDataVersioning()
    data_saver.save_versioned_dataframe(cleaned_data, "data/processed", "fiqa_df_preprocessed")

    # Get preprocessing summary
    summary = preprocessor.get_preprocessing_summary()