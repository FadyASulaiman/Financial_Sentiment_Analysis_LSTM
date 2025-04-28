import re
import pandas as pd
import spacy
from typing import List, Dict, Optional

from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.utils.loggers.feat_eng_pipeline_logger import logger
from src.lexicons.financial_lexicon import FinancialLexicon as fl


class FinancialEntityExtractor(FeatureExtractorBase):
    """
    Extract company names from text using a combination of rule-based and NER approaches.
    Works with stock tickers and company name mentions.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Parse config
        self.use_spacy = self.config.get('use_spacy', True)
        self.text_column = self.config.get('text_column', 'Sentence')
        self.output_column = self.config.get('output_column', 'Company')
        self.spacy_model = self.config.get('spacy_model', 'en_core_web_lg')
        
        self.ticker_pattern = r'\$([A-Za-z]{1,5})'  # Pattern to match stock tickers
        
        # Load spaCy model for NER
        if self.use_spacy:
            try:
                self.nlp = spacy.load(self.spacy_model)
            except:
                print(f"Installing spaCy model {self.spacy_model}...")
                import subprocess
                subprocess.call(["python", "-m", "spacy", "download", self.spacy_model])
                self.nlp = spacy.load(self.spacy_model)
        
        # Import company identifiers, known companies and stocks
        self.company_identifiers = fl.company_identifiers
        self.known_companies = fl.known_companies
        self.ticker_to_company = fl.ticker_to_company
    
    def fit(self, X, y=None):
        return self
        
    def extract_ticker(self, text: str) -> Optional[str]:
        """
        Extract stock ticker symbol from text.
        
        """
        match = re.search(self.ticker_pattern, text)
        if match:
            ticker = match.group(1)
            # Return company name if we know the ticker
            if ticker in self.ticker_to_company:
                return self.ticker_to_company[ticker]
            return ticker
        return None
    
    def extract_company_from_text(self, text: str) -> Optional[str]:
        """
        Extract company names using rules and known company names.
        This is a fallback for when NER doesn't work.
        
        """
        # Check for known companies first
        for company, full_name in self.known_companies.items():
            if company in text:
                # Make sure it's not part of another word
                pattern = r'\b' + re.escape(company) + r'\b'
                if re.search(pattern, text):
                    return full_name
        
        # Look for common company identifiers
        for identifier in self.company_identifiers:
            pattern = r'([A-Z][A-Za-z\-\s]+)\s+' + re.escape(identifier) + r'\b'
            match = re.search(pattern, text)
            if match:
                return f"{match.group(1)} {identifier}"
        
        return None
    
    def extract_company_spacy(self, text: str) -> Optional[str]:
        """Extract company names using spaCy NER."""
        doc = self.nlp(text)
        
        # Extract all ORG entities
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        
        if orgs:
            # Return the first organization found
            return orgs[0]
        
        return None
    
    def extract_company(self, text: str) -> str:
        """Main method to extract company name using all available methods."""
        # First try to extract ticker (highest precision)
        company = self.extract_ticker(text)
        if company:
            return company
        
        # Then try spaCy NER
        if self.use_spacy:
            company = self.extract_company_spacy(text)
            if company:
                return company
        
        # Fallback to rule-based extraction
        company = self.extract_company_from_text(text)
        if company:
            return company
        
        # No company found
        return "None"
    
    def transform(self, X):
        """Add a company column to the DataFrame."""
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            X_transformed[self.output_column] = X_transformed[self.text_column].apply(self.extract_company)
            return X_transformed
        else:
            # Handle case where X is a Series or a list of strings
            if isinstance(X, pd.Series):
                texts = X.values
            else:
                texts = X
            
            results = [self.extract_company(text) for text in texts]
            
            if isinstance(X, pd.Series):
                return pd.Series(results, index=X.index)
            else:
                return results
    
    def get_feature_names(self) -> List[str]:
        return [self.output_column]