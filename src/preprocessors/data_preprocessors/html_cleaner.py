from bs4 import BeautifulSoup
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.loggers.preprocessing_logger import logger

class HTMLCleaner(PreprocessorBase):
    """Class to remove HTML tags from text"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Remove HTML tags using BeautifulSoup"""
        logger.info("Removing HTML tags")
        return X.apply(lambda text: BeautifulSoup(str(text), 'html.parser').get_text())