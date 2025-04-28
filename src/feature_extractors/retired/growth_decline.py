import re
import pandas as pd
from typing import Dict

from src.feature_extractors.extractor_base import FeatureExtractorBase

from src.utils.feat_eng_pipeline_logger import logger

class GrowthDeclineQuantifier(FeatureExtractorBase):
    """Quantify growth and decline language in financial texts"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Patterns for growth language
        self.growth_patterns = [
            r'\b(?:grew|grown|growing|grow)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'\b(?:increased|increasing|increase)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'\b(?:rise|rising|risen|rose)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'\b(?:up|higher|jumped|gained|expanded)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s+(?:increase|growth|gain|jump|rise)',
            r'(?:growth|increase)\s+of\s+(\d+(?:\.\d+)?)\s*%'
        ]
        
        # Patterns for decline language
        self.decline_patterns = [
            r'\b(?:fell|fallen|falling|fall)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'\b(?:decreased|decreasing|decrease)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'\b(?:decline|declining|declined)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'\b(?:down|lower|dropped|lost|contracted)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s+(?:decrease|decline|drop|loss|contraction)',
            r'(?:decline|decrease)\s+of\s+(\d+(?:\.\d+)?)\s*%'
        ]
        
        # Compile patterns
        self.compiled_growth = [re.compile(pattern, re.IGNORECASE) for pattern in self.growth_patterns]
        self.compiled_decline = [re.compile(pattern, re.IGNORECASE) for pattern in self.decline_patterns]
        
        # Additional pattern for extracting any percentage
        self.percentage_pattern = re.compile(r'(\+|\-)?(\d+(?:\.\d+)?)\s*%', re.IGNORECASE)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            features = pd.DataFrame(index=X.index)
            
            # Extract growth percentages
            def extract_growth_values(text):
                values = []
                for pattern in self.compiled_growth:
                    matches = pattern.findall(text)
                    if matches:
                        values.extend([float(match) if isinstance(match, str) else float(match[0]) for match in matches])
                return values
            
            # Extract decline percentages
            def extract_decline_values(text):
                values = []
                for pattern in self.compiled_decline:
                    matches = pattern.findall(text)
                    if matches:
                        values.extend([float(match) if isinstance(match, str) else float(match[0]) for match in matches])
                return values
            
            # Extract any percentages
            def extract_all_percentages(text):
                matches = self.percentage_pattern.findall(text)
                if matches:
                    return [float(f"{sign if sign else ''}{value}") for sign, value in matches]
                return []
            
            # Calculate features
            growth_values = X['Sentence'].apply(extract_growth_values)
            decline_values = X['Sentence'].apply(extract_decline_values)
            all_percentages = X['Sentence'].apply(extract_all_percentages)
            
            # Max growth value
            features['max_growth_pct'] = growth_values.apply(
                lambda x: max(x) if x else 0
            )
            
            # Max decline value
            features['max_decline_pct'] = decline_values.apply(
                lambda x: max(x) if x else 0
            )
            
            # Growth mention counts
            features['growth_mention_count'] = growth_values.apply(len)
            
            # Decline mention counts
            features['decline_mention_count'] = decline_values.apply(len)
            
            # Has any growth mention
            features['has_growth_mention'] = features['growth_mention_count'] > 0
            
            # Has any decline mention
            features['has_decline_mention'] = features['decline_mention_count'] > 0
            
            # Average percentage mentioned (including both growth and decline)
            features['avg_percentage_value'] = all_percentages.apply(
                lambda x: sum(x) / len(x) if x else 0
            )
            
            # Max absolute percentage mentioned
            features['max_abs_percentage'] = all_percentages.apply(
                lambda x: max([abs(val) for val in x]) if x else 0
            )
            
            # Percentage sentiment (positive vs negative percentages)
            features['percentage_sentiment'] = all_percentages.apply(
                lambda x: sum(1 for val in x if val > 0) - sum(1 for val in x if val < 0) if x else 0
            )
            
            return pd.concat([X, features], axis=1)
            
        except Exception as e:
            logger.error(f"Error in GrowthDeclineQuantifier: {str(e)}")
            # Return empty DataFrame with same index in case of error
            return X
    
    def get_feature_names(self):
        return [
            'max_growth_pct', 'max_decline_pct',
            'growth_mention_count', 'decline_mention_count',
            'has_growth_mention', 'has_decline_mention',
            'avg_percentage_value', 'max_abs_percentage',
            'percentage_sentiment'
        ]