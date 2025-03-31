import re
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger
from src.utils.constants import SPECIAL_TOKENS

class NumericNormalizer(PreprocessorBase):
    """Class to normalize numeric expressions"""

    def __init__(self):
        # Define patterns
        self.decimal_pattern = re.compile(r'(\d+)\s+(\.\s*)(\d+)')
        self.large_number_pattern = re.compile(r'(\d+)\s*,\s*(\d+)')
        self.percentage_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:percent|%)')
        self.spaced_digits_pattern = re.compile(r'(\d+)(?:\s+(\d+))+')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Normalize numeric expressions"""
        logger.info("Normalizing numeric expressions")
        
        # Fix decimal points
        X = X.apply(lambda text: self.decimal_pattern.sub(r'\1.\3', str(text)))
        # Fix large numbers
        X = X.apply(lambda text: self.large_number_pattern.sub(r'\1,\2', str(text)))
        # Handle percentages
        X = X.apply(lambda text: self._convert_percentages(text))
        # Handle spaced digits
        X = X.apply(lambda text: self._fix_spaced_digits(text))

        return X

    def _convert_percentages(self, text):
        """Convert percentage expressions to standard form"""
        def replace_percent(match):
            number = match.group(1)
            try:
                number = float(number)
                return f"{SPECIAL_TOKENS['PERCENT']}{int(number)}"
            except ValueError:
                return match.group(0)
        
        return self.percentage_pattern.sub(replace_percent, str(text))

    def _fix_spaced_digits(self, text):
        """Remove spaces between digits in numbers"""
        def replace_spaced_digits(match):
            # Combine all digit groups by removing spaces
            full_match = match.group(0)
            return ''.join(full_match.split())
        
        return self.spaced_digits_pattern.sub(replace_spaced_digits, str(text))