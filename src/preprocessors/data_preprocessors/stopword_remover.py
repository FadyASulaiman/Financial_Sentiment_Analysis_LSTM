from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger

class StopWordRemover(PreprocessorBase):
    """Class to remove stop words"""
    
    def __init__(self, domain_specific_stopwords=None):
        self.domain_specific_stopwords = domain_specific_stopwords or []
        
        # Load standard English stop words
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        except ImportError:
            logger.warning("NLTK not available, using a minimal set of stop words")
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'of', 'on', 'to', 'with','s'}
        
        # Add domain-specific stop words
        self.stop_words.update(self.domain_specific_stopwords)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Remove stop words"""
        logger.info("Removing stop words")
        
        def remove_stops(text):
            # Tokenize
            tokens = text.split()
            # Remove stop words
            filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
            # Join back to text
            return ' '.join(filtered_tokens)
        
        X = X.apply(remove_stops)

        cleaned_text_df = X.copy()
        cleaned_text_df.to_csv('data/cleaned_text.csv', index=False)
        logger.info("Cleaned text (pre-tokenization) saved to data/cleaned_text.csv")
        
        return X