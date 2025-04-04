import re
import pandas as pd

from typing import Dict

from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.lexicons.financial_lexicon import FinancialLexicon

from src.utils.feat_eng_pipeline_logger import logger



class FinancialPerformanceIndicatorExtractor(FeatureExtractorBase):
    """Extract financial performance indicators"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.indicators = FinancialLexicon.PERFORMANCE_INDICATORS
        
        # Compile regex patterns for each indicator
        self.indicator_patterns = {}
        for indicator, terms in self.indicators.items():
            pattern = r'\b(?:' + '|'.join(terms) + r')\b'
            self.indicator_patterns[indicator] = re.compile(pattern, re.IGNORECASE)
        
        # Pattern to identify earnings context
        self.earnings_context = re.compile(
            r'\b(?:earnings|revenue|profit|income|sales|results|quarterly|performance|financials)\b',
            re.IGNORECASE
        )
        
        # Pattern to identify guidance context
        self.guidance_context = re.compile(
            r'\b(?:guidance|forecast|outlook|projection|expect|anticipate|future|next quarter|next year)\b',
            re.IGNORECASE
        )
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            features = pd.DataFrame(index=X.index)
            
            # Extract binary indicator features
            for indicator, pattern in self.indicator_patterns.items():
                features[f'has_{indicator}'] = X['Sentence'].apply(
                    lambda x: 1 if pattern.search(x) else 0
                )
            
            # Check for indicators in appropriate context
            def has_earnings_beat_in_context(text):
                if not self.indicator_patterns['earnings_beat'].search(text):
                    return 0
                
                if self.earnings_context.search(text):
                    return 1
                return 0
            
            def has_earnings_miss_in_context(text):
                if not self.indicator_patterns['earnings_miss'].search(text):
                    return 0
                
                if self.earnings_context.search(text):
                    return 1
                return 0
            
            def has_guidance_up_in_context(text):
                if not self.indicator_patterns['guidance_up'].search(text):
                    return 0
                
                if self.guidance_context.search(text):
                    return 1
                return 0
            
            def has_guidance_down_in_context(text):
                if not self.indicator_patterns['guidance_down'].search(text):
                    return 0
                
                if self.guidance_context.search(text):
                    return 1
                return 0
            
            features['earnings_beat_in_context'] = X['Sentence'].apply(has_earnings_beat_in_context)
            features['earnings_miss_in_context'] = X['Sentence'].apply(has_earnings_miss_in_context)
            features['guidance_up_in_context'] = X['Sentence'].apply(has_guidance_up_in_context)
            features['guidance_down_in_context'] = X['Sentence'].apply(has_guidance_down_in_context)
            
            # Create composite performance score
            features['performance_score'] = (
                features['has_earnings_beat'] + features['has_guidance_up'] - 
                features['has_earnings_miss'] - features['has_guidance_down']
            )
            
            # Create contextualized performance score
            features['contextualized_performance_score'] = (
                features['earnings_beat_in_context'] + features['guidance_up_in_context'] - 
                features['earnings_miss_in_context'] - features['guidance_down_in_context']
            )
            
            return pd.concat([X, features], axis=1)
            
        except Exception as e:
            logger.error(f"Error in FinancialPerformanceIndicatorExtractor: {str(e)}")
            # Return empty DataFrame with same index in case of error
            return X
    
    def get_feature_names(self):
        return [
            'has_earnings_beat', 'has_earnings_miss', 
            'has_guidance_up', 'has_guidance_down',
            'earnings_beat_in_context', 'earnings_miss_in_context',
            'guidance_up_in_context', 'guidance_down_in_context',
            'performance_score', 'contextualized_performance_score'
        ]