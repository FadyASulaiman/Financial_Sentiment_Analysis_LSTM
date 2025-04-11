import os
import spacy
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger

class SpacyLemmatizer(PreprocessorBase):
    """Class to lemmatize text using spaCy's financial domain model"""
    
    def __init__(self, model="en_core_web_sm"):
        self.model_name = model
        self.nlp = None
    
    def fit(self, X, y=None):
        """Load the spaCy model"""
        logger.info(f"Loading spaCy model {self.model_name}")
        try:
            self.nlp = spacy.load(self.model_name)
            # Disable unnecessary components for better performance
            self.nlp.disable_pipe("parser")
            self.nlp.disable_pipe("ner")
        except OSError:
            logger.warning(f"spaCy model {self.model_name} not found. Downloading...")
            os.system(f"python -m spacy download {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            self.nlp.disable_pipe("parser")
            self.nlp.disable_pipe("ner")
        return self
    
    def transform(self, X):
        """Lemmatize text using spaCy"""
        logger.info("Lemmatizing text using spaCy")
        
        if self.nlp is None:
            self.fit(X)
        
        # Function to lemmatize a text
        def lemmatize(tokens):
            if not tokens:
                return []
            
            # Join tokens to process with spaCy
            text = ' '.join(tokens)
            
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract lemmas
            lemmas = [token.lemma_ for token in doc]
            
            return lemmas
        
        # Apply lemmatization to each tokenized text
        lemmatized = X.apply(lemmatize)
        
        return lemmatized