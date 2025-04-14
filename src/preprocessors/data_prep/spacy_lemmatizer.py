import os
import pandas as pd
import spacy
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.loggers.preprocessing_logger import logger

class SpacyLemmatizer(PreprocessorBase):
    """Class responsible for lemmatization using spaCy"""
    
    def __init__(self):
        """Initialize lemmatizer with spaCy"""
        logger.info("Loading spaCy model for lemmatization")

        self.nlp = None
    
    def fit(self, X, y=None):
        """Load spaCy model on first fit"""
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            # Increase max_length for longer financial headlines
            self.nlp.max_length = 1500000
        return self
    
    def transform(self, X: pd.Series) -> pd.Series:
        """Lemmatize tokenized texts
        
        Args:
            X: pandas.Series where each element is a list of tokens
            
        Returns:
            pandas.Series of lemmatized texts (each element is a list of lemmas)
        """
        logger.info("Lemmatizing text")
        
        if self.nlp is None:
            raise ValueError("Lemmatizer not fitted. Call fit() first.")
        
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        
        lemmatized_texts = X.apply(self._lemmatize_tokens)
        
        # Log example
        if not lemmatized_texts.empty:
            logger.info(f"Example - Original tokens: {X.iloc[0][:10]}")
            logger.info(f"Example - Lemmatized: {lemmatized_texts.iloc[0][:10]}")
            
        return lemmatized_texts
    
    def _lemmatize_tokens(self, tokens):
        """Lemmatize a list of tokens"""
        # Join tokens into text for spaCy processing
        text = ' '.join(tokens)
        doc = self.nlp(text)
        
        # Extract lemmas but preserve financial symbols
        lemmas = []
        for token in doc:
            if token.text in ['$', '%'] or token.text.isdigit():
                lemmas.append(token.text)
            else:
                lemmas.append(token.lemma_)
        
        return lemmas