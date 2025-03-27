import re
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger
from utils.constants import SPECIAL_TOKENS

class CurrencyReplacer(PreprocessorBase):
    """Class to replace currency symbols/abbreviations with a special token"""

    def __init__(self):
        # Currency pattern
        self.currency_pattern = re.compile(r'(?:\$|£|€|EUR|USD|GBP|AUD|JPY|CHF|CAD)\s*(?=\d)')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Replace currency symbols/abbreviations with CUR token"""
        logger.info("Replacing currency symbols/abbreviations")
        return X.apply(lambda text: self.currency_pattern.sub(SPECIAL_TOKENS['CUR'], str(text)))