import pandas as pd
from transformers import BertTokenizer
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger
from utils.constants import SPECIAL_TOKENS

class FinBERTTokenizer(PreprocessorBase):
    """Class to tokenize text using FinBERT tokenizer"""
    
    def __init__(self, finbert_model="ProsusAI/finbert"):
        self.finbert_model = finbert_model
        self.tokenizer = None
    
    def fit(self, X, y=None):
        """Load the tokenizer"""
        logger.info(f"Loading FinBERT tokenizer from {self.finbert_model}")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.finbert_model)
        except Exception as e:
            logger.error(f"Failed to load FinBERT tokenizer: {e}")
            # Fall back to a standard BERT tokenizer
            logger.info("Falling back to standard BERT tokenizer")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return self
    
    def transform(self, X):
        """Tokenize text using FinBERT tokenizer"""
        logger.info("Tokenizing text using FinBERT tokenizer")
        
        if self.tokenizer is None:
            self.fit(X)
        
        # Tokenize each text
        tokenized = []
        for text in X:
            # Add special tokens at the start and end
            text = f"{SPECIAL_TOKENS['START']} {text} {SPECIAL_TOKENS['END']}"
            
            # Tokenize
            tokens = self.tokenizer.tokenize(text)
            tokenized.append(tokens)
        
        return pd.Series(tokenized)