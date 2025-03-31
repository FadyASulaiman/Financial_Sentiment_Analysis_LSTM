import re
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger

class SpecialCharRemover(PreprocessorBase):
    """Class to remove special characters except when part of financial notation"""
    
    def __init__(self):
        # This pattern excludes [$, %, ., !, ?, :, ;, ,] (handled separately)
        self.special_char_pattern = re.compile(r'[^\w\s$%\.!?;:,]')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Remove special characters"""
        logger.info("Removing special characters")
        return X.apply(lambda text: self.special_char_pattern.sub(' ', str(text)))