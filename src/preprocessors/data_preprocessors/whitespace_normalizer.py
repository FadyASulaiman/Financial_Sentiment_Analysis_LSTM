import re
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.loggers.preprocessing_logger import logger

class WhitespaceNormalizer(PreprocessorBase):
    """Class to normalize whitespace"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Normalize whitespace"""
        logger.info("Normalizing whitespace")
        # Replace multiple spaces with single space
        X = X.apply(lambda text: re.sub(r'\s+', ' ', str(text)))
        # Trim leading and trailing whitespace
        X = X.apply(lambda text: text.strip())
        return X