import re
import pandas as pd
from typing import Dict

from src.feature_extractors.extractor_base import FeatureExtractorBase

from src.utils.feat_eng_pipeline_logger import logger

class RelativeChangeExtractor(FeatureExtractorBase):
    """Extract relative change indicators from text"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.up_patterns = [
            r'\b(?:up|increase|rise|gain|grew|higher|climb|jumped|grew|surged|improved)\b',
            r'\b(?:growth|expansion|improvement|rally|recovery)\b',
            r'\b(?:\+\d+|\d+\s*\%\s*(?:increase|up|higher))\b'
        ]
        self.down_patterns = [
            r'\b(?:down|decrease|fall|drop|decline|lower|fell|slipped|decreased|reduced)\b',
            r'\b(?:loss|contraction|reduction|slump|downturn)\b',
            r'\b(?:-\d+|\d+\s*\%\s*(?:decrease|down|lower))\b'
        ]
        self.compiled_up = [re.compile(pattern, re.IGNORECASE) for pattern in self.up_patterns]
        self.compiled_down = [re.compile(pattern, re.IGNORECASE) for pattern in self.down_patterns]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            # Extract change indicators and surrounding context
            features = pd.DataFrame(index=X.index)
            
            # Count matches for up/down patterns
            features['rel_change_up_count'] = X['Sentence'].apply(
                lambda x: sum(1 for pattern in self.compiled_up if pattern.search(x))
            )
            features['rel_change_down_count'] = X['Sentence'].apply(
                lambda x: sum(1 for pattern in self.compiled_down if pattern.search(x))
            )
            
            # Calculate ratio of up to down indicators
            features['rel_change_ratio'] = features.apply(
                lambda row: row['rel_change_up_count'] / max(1, row['rel_change_up_count'] + row['rel_change_down_count']),
                axis=1
            )
            
            # Binary features indicating presence of change patterns
            features['has_up_indicators'] = features['rel_change_up_count'] > 0
            features['has_down_indicators'] = features['rel_change_down_count'] > 0
            
            # Context-based feature: Does the change apply to a financial metric?
            financial_metric_patterns = [
                r'\b(?:revenue|sales|profit|margin|earnings|eps|income|growth)\b',
                r'\b(?:price|value|volume|share|stock|rate|ratio)\b'
            ]
            compiled_metrics = [re.compile(pattern, re.IGNORECASE) for pattern in financial_metric_patterns]
            
            def extract_change_context(text):
                # Find all matches of change indicators
                up_matches = [m for pattern in self.compiled_up for m in pattern.finditer(text)]
                down_matches = [m for pattern in self.compiled_down for m in pattern.finditer(text)]
                all_matches = up_matches + down_matches
                
                # For each match, check if there's a financial metric nearby
                context_count = 0
                for match in all_matches:
                    # Look at a window of 5 words before and after the match
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end]
                    
                    # Check if any financial metric is in this context
                    if any(metric.search(context) for metric in compiled_metrics):
                        context_count += 1
                
                return context_count
            
            features['financial_change_context'] = X['Sentence'].apply(extract_change_context)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in RelativeChangeExtractor: {str(e)}")
            # Return empty DataFrame with same index in case of error
            return pd.DataFrame(index=X.index)
        
    def get_feature_names(self):
        return [
            'rel_change_up_count',
            'rel_change_down_count',
            'rel_change_ratio',
            'has_up_indicators',
            'has_down_indicators',
            'financial_change_context'
        ]