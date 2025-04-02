import re
import pandas as pd

from typing import Dict

from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.lexicons.financial_lexicon import FinancialLexicon

from src.utils.feat_eng_pipeline_logger import logger


class IndustrySectorCategorizer(FeatureExtractorBase):
    """Categorize texts into industries and sectors"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.industries = FinancialLexicon.INDUSTRIES
        
        # Prepare industry patterns
        self.industry_patterns = {}
        for industry, terms in self.industries.items():
            pattern = r'\b(?:' + '|'.join([re.escape(term) for term in terms]) + r')\b'
            self.industry_patterns[industry] = re.compile(pattern, re.IGNORECASE)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            features = pd.DataFrame(index=X.index)
            
            # For each industry, create a binary feature
            for industry, pattern in self.industry_patterns.items():
                features[f'industry_{industry}'] = X['Sentence'].apply(
                    lambda x: 1 if pattern.search(x) else 0
                )
            
            # Count total industry mentions
            features['industry_mention_count'] = features[[f'industry_{ind}' for ind in self.industries]].sum(axis=1)
            
            # Primary industry (the one with most mentions, or first in case of ties)
            def get_primary_industry(row):
                industry_cols = [f'industry_{ind}' for ind in self.industries]
                if row[industry_cols].sum() == 0:
                    return 'unknown'
                # Get the industry with highest value (or first in case of ties)
                primary_idx = row[industry_cols].values.argmax()
                return list(self.industries.keys())[primary_idx]
            
            features['primary_industry'] = features.apply(get_primary_industry, axis=1)
            
            # One-hot encode primary industry
            primary_dummies = pd.get_dummies(features['primary_industry'], prefix='primary')
            features = pd.concat([features, primary_dummies], axis=1)
            
            # Group industries into broader sectors
            tech_finance = ['technology', 'finance', 'telecom', 'media']
            industrial_energy = ['industrial', 'energy', 'materials', 'automotive']
            consumer_health = ['consumer', 'healthcare', 'realestate']
            
            features['sector_tech_finance'] = features[[f'industry_{ind}' for ind in tech_finance]].max(axis=1)
            features['sector_industrial_energy'] = features[[f'industry_{ind}' for ind in industrial_energy]].max(axis=1)
            features['sector_consumer_health'] = features[[f'industry_{ind}' for ind in consumer_health]].max(axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in IndustrySectorCategorizer: {str(e)}")
            # Return empty DataFrame with same index in case of error
            return pd.DataFrame(index=X.index)
    
    def get_feature_names(self):
        industry_features = [f'industry_{ind}' for ind in self.industries]
        primary_features = [f'primary_{ind}' for ind in list(self.industries.keys()) + ['unknown']]
        sector_features = ['sector_tech_finance', 'sector_industrial_energy', 'sector_consumer_health']
        return industry_features + ['industry_mention_count', 'primary_industry'] + primary_features + sector_features