import re
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger
from src.utils.constants import SPECIAL_TOKENS

class DateTimeNormalizer(PreprocessorBase):
    """Class to normalize date and time expressions"""
    
    def __init__(self):
        # Simple date pattern (e.g., January 15, 2023) - Needs improvement to include other formats
        self.date_pattern = re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b')
        
        # Time pattern (e.g., 3:37 p.m.)
        self.time_pattern = re.compile(r'\b\d{1,2}:\d{2}(?:\s*[ap]\.m\.)\b')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Normalize date and time expressions"""
        logger.info("Normalizing date and time expressions")

        # Replace dates with DATE token
        X = X.apply(lambda text: self.date_pattern.sub(SPECIAL_TOKENS['DATE'], str(text)))

        # Replace times with TIME token
        X = X.apply(lambda text: self.time_pattern.sub(SPECIAL_TOKENS['TIME'], str(text)))

        return X