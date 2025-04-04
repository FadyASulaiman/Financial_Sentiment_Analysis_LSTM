from typing import Dict

import pandas as pd
from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.feature_extractors.financial_entity import FinancialEntityExtractor

class CompositeFeatureExtractor(FeatureExtractorBase):
    """Creates high-level composite features from other extractors' output"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.transformers = []
        
        # Create smaller versions of the main extractors
        self.sentiment_extractor = None
        self.entity_extractor = FinancialEntityExtractor(config)
        self.event_extractor = None
        
        self.transformers = [
            self.sentiment_extractor,
            self.entity_extractor,
            self.event_extractor
        ]
        
    def fit(self, X, y=None):
        for transformer in self.transformers:
            transformer.fit(X, y)
        return self
    
    def transform(self, X):
        # Get features from individual extractors
        sentiment_features = self.sentiment_extractor.transform(X)
        entity_features = self.entity_extractor.transform(X)
        event_features = self.event_extractor.transform(X)
        
        # Create composite features dataframe
        composite_features = pd.DataFrame(index=X.index)
        
        # 1. Sentiment-Entity interaction features
        if not sentiment_features.empty and not entity_features.empty:
            if 'finvader_compound' in sentiment_features.columns and 'has_company' in entity_features.columns:
                composite_features['company_sentiment'] = (
                    sentiment_features['finvader_compound'] * entity_features['has_company']
                )
            
            if 'finvader_compound' in sentiment_features.columns and 'entity_richness' in entity_features.columns:
                composite_features['entity_rich_sentiment'] = (
                    sentiment_features['finvader_compound'] * 
                    entity_features['entity_richness'].clip(0, 5) / 5
                )
        
        # 2. Sentiment-Event interaction features
        if not sentiment_features.empty and not event_features.empty:
            event_categories = [col for col in event_features.columns if col.startswith('category_')]
            for category in event_categories:
                if 'finvader_compound' in sentiment_features.columns:
                    composite_features[f'{category}_sentiment'] = (
                        sentiment_features['finvader_compound'] * event_features[category]
                    )
        
        # 3. Content richness score
        richness_features = []
        
        if 'entity_richness' in entity_features.columns:
            richness_features.append(entity_features['entity_richness'])
            
        if 'event_mention_count' in event_features.columns:
            richness_features.append(event_features['event_mention_count'])
            
        if richness_features:
            composite_features['content_richness'] = sum(richness_features)
        
        # 4. Sentiment certainty
        if 'finvader_pos' in sentiment_features.columns and 'finvader_neg' in sentiment_features.columns:
            composite_features['sentiment_certainty'] = (
                sentiment_features['finvader_pos'] + sentiment_features['finvader_neg']
            )
        
        # Return the composite features
        return composite_features
    
    def get_feature_names(self):
        return [
            'company_sentiment', 'entity_rich_sentiment', 
            'content_richness', 'sentiment_certainty'
        ] + [
            f'category_{cat}_sentiment' for cat in [
                'performance', 'strategic', 'operational', 'external'
            ]
        ]