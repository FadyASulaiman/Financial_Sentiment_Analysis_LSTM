import pandas as pd
from typing import Dict
import os
import re
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.feature_extractors.extractor_base import FeatureExtractorBase

from src.utils.feat_eng_pipeline_logger import logger


class FinVADERSentimentExtractor(FeatureExtractorBase):
    """Extract sentiment features using FinVADER for financial text"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        config = config or {}
        finvader_config = config.get('features', {}).get('finvader', {})
        
        # Configuration parameters
        self.use_sentence_segmentation = finvader_config.get('use_sentence_segmentation', True)
        self.use_domain_specific_lexicon = finvader_config.get('use_domain_specific_lexicon', True)
        self.lexicon_path = finvader_config.get('lexicon_path', None)
        self.threshold_neutral = finvader_config.get('threshold_neutral', 0.05)
        
        # Initialize the analyzer
        self._initialize_analyzer()
        
    def _initialize_analyzer(self):
        """Initialize the VADER sentiment analyzer with FinVADER lexicon if available"""
        try:
            # Ensure nltk vader lexicon is downloaded
            nltk.download('vader_lexicon', quiet=True)
            
            # Create the base analyzer
            self.analyzer = SentimentIntensityAnalyzer()
            
            # Add FinVADER lexicon if requested
            if self.use_domain_specific_lexicon:
                self._add_financial_lexicon()
                
            logger.info("FinVADER sentiment analyzer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing FinVADER analyzer: {str(e)}")
            self.analyzer = None
    
    def _add_financial_lexicon(self):
        """Add financial domain-specific lexicon to the VADER sentiment analyzer"""
        
        # Check if custom lexicon path is provided
        if self.lexicon_path and os.path.exists(self.lexicon_path):
            try:
                with open(self.lexicon_path, 'r') as f:
                    financial_lexicon = json.load(f)
                    
                # Update the lexicon
                self.analyzer.lexicon.update(financial_lexicon)
                logger.info(f"Loaded custom financial lexicon from {self.lexicon_path}")
                return
            except Exception as e:
                logger.error(f"Error loading custom lexicon: {str(e)}")
        
        # Update the lexicon
        self.analyzer.lexicon.update(financial_lexicon)
        logger.info("Added built-in financial terms to lexicon")
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            if self.analyzer is None:
                logger.error("Sentiment analyzer not initialized")
                return pd.DataFrame(index=X.index)
            
            features = pd.DataFrame(index=X.index)
            
            # Apply sentiment analysis to each text
            def analyze_sentiment(text):
                # Basic sentiment scores
                vader_scores = self.analyzer.polarity_scores(text)
                
                # Additional financial context analysis
                financial_context = self._get_financial_context(text)
                
                # Combined results
                results = {**vader_scores, **financial_context}
                return results
            
            # Extract sentence-level and document-level sentiment
            if self.use_sentence_segmentation:
                def process_with_sentences(text):
                    # Split into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    
                    # Get scores for each sentence
                    sentence_scores = [self.analyzer.polarity_scores(sentence) for sentence in sentences]
                    
                    # Calculate aggregate statistics
                    result = {
                        'compound': sum(score['compound'] for score in sentence_scores) / len(sentence_scores),
                        'pos': sum(score['pos'] for score in sentence_scores) / len(sentence_scores),
                        'neu': sum(score['neu'] for score in sentence_scores) / len(sentence_scores),
                        'neg': sum(score['neg'] for score in sentence_scores) / len(sentence_scores),
                        'sentence_count': len(sentences),
                        'max_compound': max(score['compound'] for score in sentence_scores),
                        'min_compound': min(score['compound'] for score in sentence_scores),
                        'sentiment_variance': sum((score['compound'] - sum(s['compound'] for s in sentence_scores)/len(sentence_scores))**2 
                                              for score in sentence_scores) / len(sentence_scores),
                        'positive_sentence_ratio': sum(1 for score in sentence_scores if score['compound'] > self.threshold_neutral) / len(sentences),
                        'negative_sentence_ratio': sum(1 for score in sentence_scores if score['compound'] < -self.threshold_neutral) / len(sentences),
                        'neutral_sentence_ratio': sum(1 for score in sentence_scores if abs(score['compound']) <= self.threshold_neutral) / len(sentences),
                    }
                    
                    # Add financial context
                    financial_context = self._get_financial_context(text)
                    result.update(financial_context)
                    
                    return result
                
                # Apply to each document
                sentiment_data = X['Sentence'].apply(
                    lambda x: process_with_sentences(x) if isinstance(x, str) and len(x.strip()) > 0 else 
                               {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0, 'sentence_count': 0,
                                'max_compound': 0, 'min_compound': 0, 'sentiment_variance': 0,
                                'positive_sentence_ratio': 0, 'negative_sentence_ratio': 0, 'neutral_sentence_ratio': 1,
                                'financial_terms_count': 0, 'financial_sentiment_impact': 0}
                )
            else:
                # Apply sentiment analysis to each document
                sentiment_data = X['Sentence'].apply(
                    lambda x: analyze_sentiment(x) if isinstance(x, str) and len(x.strip()) > 0 else 
                               {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0,
                                'financial_terms_count': 0, 'financial_sentiment_impact': 0}
                )
            
            # Convert results to features
            for metric in sentiment_data.iloc[0].keys():
                features[f'finvader_{metric}'] = sentiment_data.apply(lambda x: x.get(metric, 0))
            
            # Add derived features
            features['finvader_sentiment_class'] = features['finvader_compound'].apply(
                lambda x: 'positive' if x > self.threshold_neutral else 
                         ('negative' if x < -self.threshold_neutral else 'neutral')
            )
            
            # Convert to one-hot encoding
            sentiment_dummies = pd.get_dummies(features['finvader_sentiment_class'], prefix='finvader')
            features = pd.concat([features, sentiment_dummies], axis=1)
            
            # Calculate the impact of financial terms on the sentiment
            features['finvader_financial_impact_ratio'] = features.apply(
                lambda row: row['finvader_financial_sentiment_impact'] / (abs(row['finvader_compound']) + 0.0001),
                axis=1
            )
            
            return pd.concat([X, features], axis=1)
            
        except Exception as e:
            logger.error(f"Error in FinVADERSentimentExtractor: {str(e)}")
            # Return empty DataFrame with same index in case of error
            return X
    
    def _get_financial_context(self, text: str) -> Dict[str, float]:
        """Extract financial context and its impact on sentiment"""
        if not text or not isinstance(text, str):
            return {'financial_terms_count': 0, 'financial_sentiment_impact': 0}
        
        # List of financial terms to look for
        financial_terms = [term for term in self.analyzer.lexicon.keys() 
                          if isinstance(self.analyzer.lexicon[term], (int, float)) and 
                          abs(self.analyzer.lexicon[term]) > 0.5]
        
        # Count financial terms
        term_count = 0
        sentiment_impact = 0
        
        for term in financial_terms:
            # Look for the term using word boundaries to match whole words
            count = len(re.findall(r'\b' + re.escape(term) + r'\b', text.lower()))
            if count > 0:
                term_count += count
                sentiment_impact += count * self.analyzer.lexicon[term]
        
        return {
            'financial_terms_count': term_count,
            'financial_sentiment_impact': sentiment_impact
        }
        
    def get_feature_names(self):
        """Return the list of feature names that this extractor produces"""
        base_features = [
            'finvader_compound', 'finvader_pos', 'finvader_neu', 'finvader_neg',
            'finvader_financial_terms_count', 'finvader_financial_sentiment_impact',
            'finvader_financial_impact_ratio',
            'finvader_sentiment_class',
            'finvader_positive', 'finvader_negative', 'finvader_neutral'
        ]
        
        # Add sentence-level features if sentence segmentation is used
        if self.use_sentence_segmentation:
            sentence_features = [
                'finvader_sentence_count', 'finvader_max_compound', 'finvader_min_compound',
                'finvader_sentiment_variance', 'finvader_positive_sentence_ratio',
                'finvader_negative_sentence_ratio', 'finvader_neutral_sentence_ratio'
            ]
            return base_features + sentence_features
        
        return base_features