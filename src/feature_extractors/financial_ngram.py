import pandas as pd
from typing import Dict, List
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.lexicons.financial_lexicon import FinancialLexicon

from src.utils.feat_eng_pipeline_logger import logger


class FinancialNGramExtractor(FeatureExtractorBase):
    """Extract financial n-grams based on a financial lexicon"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        config = config or {}
        feature_config = config.get('features', {})
        self.n_gram_range = feature_config.get('n_gram_range', (1, 3))
        self.min_df = feature_config.get('min_df', 5)
        self.max_df = feature_config.get('max_df', 0.9)
        self.max_features = feature_config.get('tfidf_max_features', 5000)
        
        # Create financial term vocabulary from lexicon
        self.financial_terms = (
            list(FinancialLexicon.POSITIVE_TERMS) + 
            list(FinancialLexicon.NEGATIVE_TERMS) + 
            list(FinancialLexicon.COMPANIES) + 
            list(FinancialLexicon.MARKET_INDICATORS)
        )
        
        self.financial_vectorizer = CountVectorizer(
            vocabulary=self.financial_terms,
            binary=True
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.n_gram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features
        )
        
    def fit(self, X, y=None):
        try:
            # Fit the general TF-IDF vectorizer
            self.tfidf_vectorizer.fit(X['Sentence'])
            # Fit the financial term vectorizer
            self.financial_vectorizer.fit(X['Sentence'])
            return self
        except Exception as e:
            logger.error(f"Error in FinancialNGramExtractor.fit: {str(e)}")
            return self
        
    def transform(self, X):
        try:
            # Transform using the financial term vectorizer
            financial_features = self.financial_vectorizer.transform(X['Sentence'])
            financial_df = pd.DataFrame(
                financial_features.toarray(),
                columns=[f'fin_term_{term}' for term in self.financial_vectorizer.get_feature_names_out()],
                index=X.index
            )
            
            # Transform using the general TF-IDF vectorizer
            tfidf_features = self.tfidf_vectorizer.transform(X['Sentence'])
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{term}' for term in self.tfidf_vectorizer.get_feature_names_out()],
                index=X.index
            )
            
            # Combine both feature sets
            combined_df = pd.concat([financial_df, tfidf_df], axis=1)
            
            # Add aggregated features
            combined_df['positive_terms_count'] = combined_df[[f'fin_term_{term}' for term in FinancialLexicon.POSITIVE_TERMS if f'fin_term_{term}' in combined_df.columns]].sum(axis=1)
            combined_df['negative_terms_count'] = combined_df[[f'fin_term_{term}' for term in FinancialLexicon.NEGATIVE_TERMS if f'fin_term_{term}' in combined_df.columns]].sum(axis=1)
            combined_df['company_mentions_count'] = combined_df[[f'fin_term_{term}' for term in FinancialLexicon.COMPANIES if f'fin_term_{term}' in combined_df.columns]].sum(axis=1)
            combined_df['market_indicator_count'] = combined_df[[f'fin_term_{term}' for term in FinancialLexicon.MARKET_INDICATORS if f'fin_term_{term}' in combined_df.columns]].sum(axis=1)
            
            # Calculate sentiment score based on term counts
            combined_df['lexicon_sentiment_score'] = (
                combined_df['positive_terms_count'] - combined_df['negative_terms_count']
            ) / (combined_df['positive_terms_count'] + combined_df['negative_terms_count'] + 1)
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error in FinancialNGramExtractor.transform: {str(e)}")
            # Return empty DataFrame with same index in case of error
            return pd.DataFrame(index=X.index)
    
    def get_feature_names(self):
        try:
            # Get all feature names
            financial_names = [f'fin_term_{term}' for term in self.financial_vectorizer.get_feature_names_out()]
            tfidf_names = [f'tfidf_{term}' for term in self.tfidf_vectorizer.get_feature_names_out()]
            aggregated_names = [
                'positive_terms_count', 'negative_terms_count', 
                'company_mentions_count', 'market_indicator_count',
                'lexicon_sentiment_score'
            ]
            return financial_names + tfidf_names + aggregated_names
        except Exception as e:
            logger.error(f"Error in FinancialNGramExtractor.get_feature_names: {str(e)}")
            return []