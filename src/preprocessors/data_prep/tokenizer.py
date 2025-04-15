import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from src.utils.loggers.data_prep_pipeline_logger import logger

from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase


class Tokenizer(PreprocessorBase):
    """Class responsible for tokenization"""
    
    def __init__(self, remove_stopwords=False):
        """
        Initialize tokenizer
        """
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
            # Keep sentiment-bearing words
            self.sentiment_words = {'not', 'no', 'never', 'more', 'most', 'up', 'down'}
            self.stop_words = self.stop_words - self.sentiment_words
    
    def fit(self, X, y=None):
        """Fit method (stateless, just returns self)"""
        return self
    
    def transform(self, X):
        """Tokenize texts into words
        
        Args:
            X: pandas.Series or list of strings containing text data
             
        Returns:
            pandas.Series of tokenized texts (each element is a list of tokens)
        """
        logger.info("Tokenizing text")
        
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        
        tokenized_texts = X.apply(self._tokenize_text)
        
        # Log example
        if not tokenized_texts.empty:
            logger.info(f"Example - Tokenized: {tokenized_texts.iloc[0]}")
            
        return tokenized_texts
    
    def _tokenize_text(self, text):
        """Tokenize a single text string"""

        if not isinstance(text, str):
            logger.debug(f"Unexpected Instance: {text}")
            text = str(text) if text == text else ""
        
        tokens = word_tokenize(text)

        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
                
        return tokens