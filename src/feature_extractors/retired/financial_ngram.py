import ast
import re
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
        
        # Updated parameters for better feature selection
        self.n_gram_range = feature_config.get('n_gram_range', (1, 2))  # Reduced from (1,3)
        self.min_df = feature_config.get('min_df', 0.01)  # Use percentage instead of count
        self.max_df = feature_config.get('max_df', 0.8)   # More strict upper limit
        self.max_features = feature_config.get('tfidf_max_features', 200)  # Reduced from 5000
        
        # Filter out rare tokens in financial terms
        self.min_financial_term_freq = feature_config.get('min_financial_term_freq', 0.005)
        
        # Get core financial terms with meaning
        self.financial_terms = (
            list(FinancialLexicon.POSITIVE_TERMS) + 
            list(FinancialLexicon.NEGATIVE_TERMS)
        )
        
        # Add only the most common companies and indicators
        top_companies = feature_config.get('top_companies', 20)
        top_indicators = feature_config.get('top_indicators', 10)
        
        if len(FinancialLexicon.COMPANIES) > top_companies:
            self.financial_terms.extend(list(FinancialLexicon.COMPANIES)[:top_companies])
        else:
            self.financial_terms.extend(list(FinancialLexicon.COMPANIES))
            
        if len(FinancialLexicon.MARKET_INDICATORS) > top_indicators:
            self.financial_terms.extend(list(FinancialLexicon.MARKET_INDICATORS)[:top_indicators])
        else:
            self.financial_terms.extend(list(FinancialLexicon.MARKET_INDICATORS))
        
        # Initialize vectorizers as None until fit
        self.financial_vectorizer = None
        self.tfidf_vectorizer = None
        self.is_fitted = False
        
    def _fit(self, X, y=None):
        # First, calculate term frequencies to select meaningful terms
        if self.min_financial_term_freq > 0:
            # Simple term frequency count for financial terms
            term_freq = {}
            total_docs = len(X)
            
            for term in self.financial_terms:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                count = sum(1 for text in X['Sentence'] if pattern.search(text))
                term_freq[term] = count / total_docs
            
            # Filter terms based on frequency
            self.filtered_terms = [term for term, freq in term_freq.items() 
                                  if freq >= self.min_financial_term_freq]
            
            if len(self.filtered_terms) < 10:  # Ensure we have at least some terms
                # Take top 10 terms by frequency
                self.filtered_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
                self.filtered_terms = [term for term, _ in self.filtered_terms[:10]]
        else:
            self.filtered_terms = self.financial_terms
            
        logger.info(f"Selected {len(self.filtered_terms)} financial terms based on frequency")
        
        # Now initialize the vectorizers with the filtered terms
        self.financial_vectorizer = CountVectorizer(
            vocabulary=self.filtered_terms,
            binary=True
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.n_gram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features
        )
        
        # Fit the vectorizers
        self.financial_vectorizer.fit(X['Sentence'])
        self.tfidf_vectorizer.fit(X['Sentence'])
        self.is_fitted = True
        return self
        
    def _transform(self, X):
        if not self.is_fitted:
            logger.warning("FinancialNGramExtractor not fitted, fitting now...")
            self._fit(X)
        
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
        

        # Add aggregated features which are more meaningful than individual terms
        pos_terms = [f'fin_term_{term}' for term in FinancialLexicon.POSITIVE_TERMS 
                    if f'fin_term_{term}' in financial_df.columns]
        neg_terms = [f'fin_term_{term}' for term in FinancialLexicon.NEGATIVE_TERMS 
                    if f'fin_term_{term}' in financial_df.columns]
        
        aggregated_df = pd.DataFrame(index=X.index)
        
        # Aggregate financial terms into categories
        if pos_terms:
            aggregated_df['positive_terms_count'] = financial_df[pos_terms].sum(axis=1)
        else:
            aggregated_df['positive_terms_count'] = 0
            
        if neg_terms:
            aggregated_df['negative_terms_count'] = financial_df[neg_terms].sum(axis=1)
        else:
            aggregated_df['negative_terms_count'] = 0
            
        # Calculate sentiment score
        denominator = aggregated_df['positive_terms_count'] + aggregated_df['negative_terms_count'] + 1
        aggregated_df['lexicon_sentiment_score'] = (
            aggregated_df['positive_terms_count'] - aggregated_df['negative_terms_count']
        ) / denominator
        
        # Add polarity strength
        aggregated_df['sentiment_magnitude'] = (
            aggregated_df['positive_terms_count'] + aggregated_df['negative_terms_count']
        )
        
        # Calculate positive/negative ratio
        aggregated_df['pos_neg_ratio'] = aggregated_df['positive_terms_count'] / (
            aggregated_df['negative_terms_count'] + 0.5
        )
        
        # Replace extreme values
        aggregated_df['pos_neg_ratio'] = aggregated_df['pos_neg_ratio'].clip(0, 10)
        
        # Combine all features
        combined_df = pd.concat([aggregated_df, financial_df, tfidf_df], axis=1)
        
        return combined_df