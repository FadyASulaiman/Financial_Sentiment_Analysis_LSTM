import re
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.loggers.preprocessing_logger import logger

class URLRemover(PreprocessorBase):
    """Class to remove URLs from text"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Remove URLs from text"""
        logger.info("Removing URLs")
        return X.apply(lambda text: self.url_pattern.sub('', str(text)))