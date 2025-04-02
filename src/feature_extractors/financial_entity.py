import re
import pandas as pd
from typing import Dict, List
import spacy
import os

from src.utils.feat_eng_pipeline_logger import logger

from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.lexicons.financial_lexicon import FinancialLexicon



# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    logger.warning(f"Error loading spaCy model: {e}. Installing it now...")
    os.system('python -m spacy download en_core_web_sm')
    try:
        nlp = spacy.load('en_core_web_sm')
    except Exception as e2:
        logger.error(f"Failed to load spaCy model after installation: {e2}")
        nlp = None

class FinancialEntityExtractor(FeatureExtractorBase):
    """Extract financial entities using gazetteers and regex patterns"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        config = config or {}
        entity_config = config.get('features', {}).get('entity_extraction', {})
        self.use_spacy = entity_config.get('use_spacy', True)
        self.use_regex = entity_config.get('use_regex', True)
        self.use_gazetteer = entity_config.get('use_gazetteer', True)
        
        # Company name gazetteer
        self.companies = set(FinancialLexicon.COMPANIES)
        
        # Compile regex patterns
        self.company_pattern = re.compile(FinancialLexicon.COMPANY_PATTERN)
        self.ticker_pattern = re.compile(FinancialLexicon.TICKER_PATTERN)
        self.percentage_pattern = re.compile(FinancialLexicon.PERCENTAGE_PATTERN)
        self.currency_amount_pattern = re.compile(FinancialLexicon.CURRENCY_AMOUNT_PATTERN)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            features = pd.DataFrame(index=X.index)
            
            # Extract entities using regex patterns
            if self.use_regex:
                # Company names (potential matches)
                features['company_regex_count'] = X['Sentence'].apply(
                    lambda x: len(self.company_pattern.findall(x))
                )
                
                # Stock tickers
                features['ticker_count'] = X['Sentence'].apply(
                    lambda x: len(self.ticker_pattern.findall(x))
                )
                
                # Percentages
                features['percentage_mentions'] = X['Sentence'].apply(
                    lambda x: len(self.percentage_pattern.findall(x))
                )
                
                # Currency amounts
                features['currency_amount_mentions'] = X['Sentence'].apply(
                    lambda x: len(self.currency_amount_pattern.findall(x))
                )
            
            # Extract entities using gazetteer
            if self.use_gazetteer:
                features['company_gazetteer_count'] = X['Sentence'].apply(
                    lambda x: sum(1 for company in self.companies if company in x.lower())
                )
            
            # Extract entities using spaCy NER
            if self.use_spacy and nlp is not None:
                def extract_spacy_entities(text):
                    doc = nlp(text)
                    org_count = sum(1 for ent in doc.ents if ent.label_ == 'ORG')
                    money_count = sum(1 for ent in doc.ents if ent.label_ == 'MONEY')
                    percent_count = sum(1 for ent in doc.ents if ent.label_ == 'PERCENT')
                    date_count = sum(1 for ent in doc.ents if ent.label_ == 'DATE')
                    return {
                        'spacy_org_count': org_count,
                        'spacy_money_count': money_count,
                        'spacy_percent_count': percent_count,
                        'spacy_date_count': date_count
                    }
                
                spacy_entities = X['Sentence'].apply(extract_spacy_entities)
                for col in ['spacy_org_count', 'spacy_money_count', 'spacy_percent_count', 'spacy_date_count']:
                    features[col] = spacy_entities.apply(lambda x: x.get(col, 0))
            
            # Combine entity detection methods
            if self.use_gazetteer and self.use_regex:
                features['company_combined_count'] = features['company_gazetteer_count'] + features['company_regex_count']
            
            # Binary features based on entity presence
            features['has_company'] = (
                (features['company_gazetteer_count'] > 0 if self.use_gazetteer else False) | 
                (features['company_regex_count'] > 0 if self.use_regex else False)
            )
            features['has_ticker'] = features['ticker_count'] > 0 if self.use_regex else False
            features['has_percentage'] = features['percentage_mentions'] > 0 if self.use_regex else False
            features['has_currency'] = features['currency_amount_mentions'] > 0 if self.use_regex else False
            
            return features
            
        except Exception as e:
            logger.error(f"Error in FinancialEntityExtractor: {str(e)}")
            # Return empty DataFrame with same index in case of error
            return pd.DataFrame(index=X.index)
    
    def get_feature_names(self):
        names = []
        if self.use_regex:
            names.extend(['company_regex_count', 'ticker_count', 'percentage_mentions', 'currency_amount_mentions'])
        if self.use_gazetteer:
            names.append('company_gazetteer_count')
        if self.use_spacy and nlp is not None:
            names.extend(['spacy_org_count', 'spacy_money_count', 'spacy_percent_count', 'spacy_date_count'])
        if self.use_gazetteer and self.use_regex:
            names.append('company_combined_count')
        names.extend(['has_company', 'has_ticker', 'has_percentage', 'has_currency'])
        return names