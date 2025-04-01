import re
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger
from src.utils.constants import SPECIAL_TOKENS

class DateTimeNormalizer(PreprocessorBase):
    """Class to normalize date and time expressions"""
    
    def __init__(self):
        self.date_pattern = re.compile(r'\b(?:'
                                        # January 15, 2023
                                        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4}|'
                                        # ex: 23 oct, 1991
                                        r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|'
                                        # Numeric formats (MM/DD/YYYY, DD/MM/YYYY, YYYY/MM/DD, MM.DD.YYYY, DD.MM.YYYY, YYYY.MM.DD)
                                        r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|'
                                        r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}|'
                                        # Month-to-month range followed by year (e.g., "August-October 2010")
                                        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*[-–—]\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|'
                                        # Quarter notation (e.g., "Q1 2010", "Q3-Q4 2009")
                                        r'Q[1-4](?:\s*[-–—]\s*Q[1-4])?\s+\d{4}|'
                                        
                                        # 1915 feb, 25
                                        r'\d{4}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?'
                                        r')\b', 
                                        re.IGNORECASE
                                    )
            

        self.time_pattern = re.compile(
            r'\b(?:'
            # 12-hour clock with AM/PM (various formats with colon or period separator)
            r'(?:1[0-2]|0?[1-9])(?:[:.][0-5][0-9])(?:[:.][0-5][0-9])?\s*(?:[aApP]\.?[mM]\.?)|'
            r'(?:1[0-2]|0?[1-9])(?:[:.][0-5][0-9])?\s*(?:[aApP]\.?[mM]\.?)|'
            # 24-hour clock (various formats with colon or period separator)
            r'(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?|'
            # Time with timezone abbreviations
            r'(?:1[0-2]|0?[1-9])(?:[:.][0-5][0-9])(?:[:.][0-5][0-9])?\s*(?:[aApP]\.?[mM]\.?)?\s*(?:GMT|UTC|EST|CST|MST|PST|EDT|CDT|MDT|PDT|[A-Z]{3,4})|'
            r'(?:[01][0-9]|2[0-3])[:.][0-5][0-9](?:[:.][0-5][0-9])?\s*(?:GMT|UTC|EST|CST|MST|PST|EDT|CDT|MDT|PDT|[A-Z]{3,4})|'
            # Time with timezone offset
            r'(?:1[0-2]|0?[1-9])(?:[:.][0-5][0-9])(?:[:.][0-5][0-9])?\s*(?:[aApP]\.?[mM]\.?)?\s*(?:[+-][01][0-9](?::?[0-5][0-9])?)|'
            r'(?:[01][0-9]|2[0-3])[:.][0-5][0-9](?:[:.][0-5][0-9])?\s*(?:[+-][01][0-9](?::?[0-5][0-9])?)|'
            # Handle 24-hour format with AM/PM (technically incorrect but occurs in real data)
            r'(?:[01][0-9]|2[0-3])[:.][0-5][0-9](?:[:.][0-5][0-9])?\s*(?:[aApP]\.?[mM]\.?)'
            r')\b'
        )
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