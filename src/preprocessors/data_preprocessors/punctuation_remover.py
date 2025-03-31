import re
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger

class PunctuationRemover(PreprocessorBase):
    """Class to remove punctuation marks"""
    
    def __init__(self):
        self.punctuation_pattern = re.compile(r'''
                                    # Match periods that are NOT decimal points
                                    (?<!\d)[.]|[.](?!\d)|
                                    
                                    # Match commas that are NOT in numbers
                                    [,](?!\d)|(?<!\d)[,]|
                                    
                                    # Match all other punctuation
                                    [!?;:]
                                ''', re.VERBOSE)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Remove punctuation marks"""
        logger.info("Removing punctuation marks")
        return X.apply(lambda text: self.punctuation_pattern.sub(' ', str(text)))