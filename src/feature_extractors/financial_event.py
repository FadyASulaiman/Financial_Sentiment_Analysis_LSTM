import re
import pandas as pd
from typing import Dict, List

from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.lexicons.financial_lexicon import FinancialLexicon

from src.utils.feat_eng_pipeline_logger import logger

class FinancialEventClassifier(FeatureExtractorBase):
    """Classify financial events in texts"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.events = FinancialLexicon.FINANCIAL_EVENTS
        
        # Prepare event patterns
        self.event_patterns = {}
        for event, terms in self.events.items():
            pattern = r'\b(?:' + '|'.join([re.escape(term) for term in terms]) + r')\b'
            self.event_patterns[event] = re.compile(pattern, re.IGNORECASE)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            features = pd.DataFrame(index=X.index)
            
            # For each event type, create a binary feature
            for event, pattern in self.event_patterns.items():
                features[f'event_{event}'] = X['Sentence'].apply(
                    lambda x: 1 if pattern.search(x) else 0
                )
            
            # Count matches for each event type
            for event, pattern in self.event_patterns.items():
                features[f'event_{event}_count'] = X['Sentence'].apply(
                    lambda x: len(pattern.findall(x))
                )
            
            # Count total event mentions
            features['event_mention_count'] = features[[f'event_{ev}' for ev in self.events]].sum(axis=1)
            
            # Primary event (the one with most mentions)
            def get_primary_event(row):
                event_cols = [f'event_{ev}_count' for ev in self.events]
                if row[event_cols].sum() == 0:
                    return 'unknown'
                # Get the event with highest count
                primary_idx = row[event_cols].values.argmax()
                return list(self.events.keys())[primary_idx]
            
            features['primary_event'] = features.apply(get_primary_event, axis=1)
            
            # One-hot encode primary event
            primary_dummies = pd.get_dummies(features['primary_event'], prefix='primary_event')
            features = pd.concat([features, primary_dummies], axis=1)
            
            # Group events into broader categories
            performance_events = ['earnings', 'dividend']
            strategic_events = ['merger_acquisition', 'restructuring', 'market_expansion']
            operational_events = ['product_launch', 'leadership_change']
            external_events = ['regulatory', 'litigation']
            
            features['category_performance'] = features[[f'event_{ev}' for ev in performance_events]].max(axis=1)
            features['category_strategic'] = features[[f'event_{ev}' for ev in strategic_events]].max(axis=1)
            features['category_operational'] = features[[f'event_{ev}' for ev in operational_events]].max(axis=1)
            features['category_external'] = features[[f'event_{ev}' for ev in external_events]].max(axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in FinancialEventClassifier: {str(e)}")
            # Return empty DataFrame with same index in case of error
            return pd.DataFrame(index=X.index)
    
    def get_feature_names(self):
        event_features = [f'event_{ev}' for ev in self.events]
        event_count_features = [f'event_{ev}_count' for ev in self.events]
        primary_features = [f'primary_event_{ev}' for ev in list(self.events.keys()) + ['unknown']]
        category_features = [
            'category_performance', 'category_strategic', 
            'category_operational', 'category_external'
        ]
        return event_features + event_count_features + ['event_mention_count', 'primary_event'] + primary_features + category_features