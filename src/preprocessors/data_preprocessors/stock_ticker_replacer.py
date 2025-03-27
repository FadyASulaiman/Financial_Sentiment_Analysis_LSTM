import re
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger
from utils.constants import SPECIAL_TOKENS

class StockTickerReplacer(PreprocessorBase):
    """Class to replace stock tickers with a special token"""
    
    def __init__(self):
        # Stock ticker pattern (e.g., $AAPL)
        self.ticker_pattern = re.compile(r'\$[A-Z]{1,5}\b')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Replace stock tickers with STOCK token"""
        logger.info("Replacing stock tickers")
        return X.apply(lambda text: self.ticker_pattern.sub(SPECIAL_TOKENS['STOCK'], str(text)))