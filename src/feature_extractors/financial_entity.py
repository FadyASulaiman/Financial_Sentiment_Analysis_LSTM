import re
import pandas as pd
import spacy
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

from src.feature_extractors.extractor_base import FeatureExtractorBase

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
        
        # Common company identifiers
        self.company_identifiers = [
            "Inc", "Corp", "Corporation", "Ltd", "Limited", "LLC", "PLC", 
            "Company", "Co", "Group", "Holdings", "Technologies", "Systems",
            "International", "Enterprises", "Oyj", "HEL", "AB", "GmbH", "AG"
        ]
        
        # Known major companies (can be extended)
        self.known_companies = {
            "IBM": "IBM",
            "Comptel": "Comptel",
            "YIT": "YIT Corporation",
            "Outokumpu": "Outokumpu Technology",
            "Tiimari": "Tiimari",
            "Nordstjernan": "Nordstjernan",
            "EQT": "EQT",
            "Stora Enso": "Stora Enso Oyj",
            "UPM-Kymmene": "UPM-Kymmene Oyj",
            "UPM": "UPM-Kymmene Oyj",
            "Starbucks": "Starbucks",
            "SMH": "SMH",
            "ZAGG": "ZAGG",
            "Vaahto Group": "Vaahto Group",
            "St1": "St1",
            "Altona": "Altona",
            "Tulla Resources": "Tulla Resources",
            "Nokian Tyres": "Nokian Tyres",
            "Bank of America": "Bank of America",
            "BofA": "Bank of America",
            "Deutsche Bank": "Deutsche Bank",
            "ThyssenKrupp": "ThyssenKrupp",
            "United Technologies": "United Technologies Corp",
            "Otis": "Otis",
            "Schindler": "Schindler AG",
            "Kone": "Kone Oyj",
            "Talentum": "Talentum",
            "BIOC": "BIOC",
            "Ragutis": "Ragutis",
            "Olvi": "Olvi",
            "Digia": "Digia",
            "YHOO": "Yahoo",
            "QQQ": "QQQ",
            "NDX": "NDX"
        }
        
        # Create a mapping from ticker to company name
        self.ticker_to_company = {
            "SMH": "VanEck Semiconductor ETF",
            "ZAGG": "ZAGG Inc",
            "BIOC": "Biocept Inc",
            "YHOO": "Yahoo",
            "QQQ": "Invesco QQQ ETF",
            "NDX": "Nasdaq-100 Index"
        }
    
    def fit(self, X, y=None):
        return self
        
    def extract_ticker(self, text: str) -> Optional[str]:
        """
        Extract stock ticker symbol from text.
        
        Args:
            text: Input text
            
        Returns:
            Company name or ticker symbol if found, None otherwise
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
        
        Args:
            text: Input text
            
        Returns:
            Company name if found, None otherwise
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
        """
        Extract company names using spaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            Company name if found, None otherwise
        """
        doc = self.nlp(text)
        
        # Extract all ORG entities
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        
        if orgs:
            # Return the first organization found
            return orgs[0]
        
        return None
    
    def extract_company(self, text: str) -> str:
        """
        Main method to extract company name using all available methods.
        
        Args:
            text: Input text
            
        Returns:
            Company name, or "None" as a string if no company is found
        """
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
        """
        Add a company column to the DataFrame.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with an additional company column
        """
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
        """
        Get the name of the generated feature.
        
        Returns:
            List of feature names
        """
        return [self.output_column]