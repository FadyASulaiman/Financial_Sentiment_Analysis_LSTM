from collections import Counter
from dataclasses import dataclass
import time
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from textstat import textstat
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from III_feature_engineering_base import BaseFeatureExtractor


@dataclass
class LinguisticStats:
    """Enhanced linguistic statistics with financial context"""
    pos_distribution: Dict[str, float]
    sentiment_scores: Dict[str, float]
    readability_scores: Dict[str, float]
    financial_entities: Dict[str, int]

class LinguisticFeatureExtractor(BaseFeatureExtractor):
    """Improved linguistic analysis with financial context awareness"""

    def __init__(self, spacy_model: str = "en_core_web_sm", batch_size: int = 500):
        super().__init__()
        self.nlp = spacy.load(spacy_model)
        self.batch_size = batch_size
        self.financial_entities = ["ORG", "MONEY", "DATE", "PERCENT"]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LinguisticFeatureExtractor":
        self._validate_input(X, ["clean_sentence"])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._validate_input(X, ["clean_sentence"])
        
        features = []
        for doc in tqdm(self.nlp.pipe(X["clean_sentence"], batch_size=self.batch_size),
                        total=len(X), desc="Processing linguistic features"):
            features.append(self._extract_linguistic_features(doc))
            
        features_df = pd.DataFrame(features).fillna(0)
        self.stats = self._calculate_stats(features_df)
        return features_df.astype(np.float32).values

    def _extract_linguistic_features(self, doc: Doc) -> Dict:
        """Extract enhanced linguistic features with financial context"""
        sia = SentimentIntensityAnalyzer()
        
        return {
            **self._extract_pos_features(doc),
            **self._extract_sentiment_features(doc.text),
            **self._extract_readability_features(doc.text),
            **self._extract_financial_entities(doc)
        }

    def _extract_pos_features(self, doc: Doc) -> Dict:
        pos_counts = Counter([token.pos_ for token in doc])
        total = len(doc)
        return {f"pos_{k}": v/total for k, v in pos_counts.items()}

    def _extract_sentiment_features(self, text: str) -> Dict:
        scores = SentimentIntensityAnalyzer().polarity_scores(text)
        return {f"sentiment_{k}": v for k, v in scores.items()}

    def _extract_readability_features(self, text: str) -> Dict:
        return {
            "flesch": textstat.flesch_reading_ease(text),
            "fk_grade": textstat.flesch_kincaid_grade(text)
        }

    def _extract_financial_entities(self, doc: Doc) -> Dict:
        entities = Counter([ent.label_ for ent in doc.ents])
        return {f"ent_{ent}": entities.get(ent, 0) for ent in self.financial_entities}

class BusinessFeatureExtractor(BaseFeatureExtractor):
    """Optimized business feature extraction with category aggregation"""

    BUSINESS_CATEGORIES = {
        "mergers_acquisitions": ["merger", "acquisition", "takeover", "buyout"],
        "financial_metrics": ["revenue", "profit", "ebitda", "margin"],
        "market_actions": ["rally", "plunge", "volatility", "correction"]
    }

    def __init__(self, spacy_model: str = "en_core_web_sm", batch_size: int = 500):
        super().__init__()
        self.nlp = spacy.load(spacy_model)
        self.batch_size = batch_size
        self.matcher = PhraseMatcher(self.nlp.vocab)
        self._initialize_matchers()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BusinessFeatureExtractor':
        """Required fit method implementation"""
        self._validate_input(X, ["clean_sentence"])
        self._is_fitted = True
        return self

    def _initialize_matchers(self):
        for category, terms in self.BUSINESS_CATEGORIES.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            self.matcher.add(category, patterns)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        features = []
        docs = self.nlp.pipe(X["clean_sentence"], batch_size=self.batch_size)
        for doc in tqdm(docs, total=len(X), desc="Extracting business features"):
            features.append(self._extract_business_features(doc))
        return np.array(pd.DataFrame(features).fillna(0))

    def _extract_business_features(self, doc: Doc) -> Dict:
        matches = self.matcher(doc)
        counts = Counter([self.nlp.vocab.strings[match_id] for match_id, _, _ in matches])
        return {f"biz_{category}": counts.get(category, 0) for category in self.BUSINESS_CATEGORIES}
