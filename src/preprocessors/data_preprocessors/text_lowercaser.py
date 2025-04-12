from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.loggers.preprocessing_logger import logger
from src.utils.constants import SPECIAL_TOKENS

class TextLowercaser(PreprocessorBase):
    """Class to lowercase text while preserving special tokens"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Lowercase text"""
        logger.info("Lowercasing text")
        
        # Create a function to preserve special tokens while lowercasing
        def lowercase_preserve_tokens(text):
            # Create a placeholder for special tokens
            placeholders = {}
            
            # Replace special tokens with placeholders
            for token_type, token in SPECIAL_TOKENS.items():
                if token in text:
                    placeholder = f"__PLACEHOLDER_{token_type}__"
                    placeholders[placeholder] = token
                    text = text.replace(token, placeholder)
            
            # Lowercase the text
            text = text.lower()
            
            # Restore special tokens
            for placeholder, token in placeholders.items():
                text = text.replace(placeholder, token)
            
            return text
        
        return X.apply(lowercase_preserve_tokens)