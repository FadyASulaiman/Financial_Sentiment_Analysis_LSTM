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
    """Statistics for linguistic features"""
    pos_distribution: Dict[str, int]
    avg_sentence_length: float
    vocab_richness: float
    sentiment_distribution: Dict[str, float]
    complexity_scores: Dict[str, float]


class LinguisticFeatureExtractor(BaseFeatureExtractor):
    """Linguistic feature extraction from text data."""

    def __init__(self, spacy_model: str = "en_core_web_sm", batch_size: int = 1000):
        super().__init__()
        self.nlp = spacy.load(spacy_model)
        self.sia = SentimentIntensityAnalyzer()
        self.batch_size = batch_size

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LinguisticFeatureExtractor":
        """Fit the feature extractor."""
        self._validate_input(X, ["clean_sentence"])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform text data into linguistic features."""
        self._validate_input(X, ["clean_sentence"])
        if not self._is_fitted:
            raise ValueError("LinguisticFeatureExtractor must be fitted before transform")

        start_time = time.time()
        all_features = []

        for i in tqdm(range(0, len(X), self.batch_size), desc="Extracting linguistic features"):
            batch = X.iloc[i : i + self.batch_size]
            batch_features = self._process_batch(batch)
            all_features.append(batch_features)

        features_df = pd.concat(all_features, axis=0, ignore_index=True)
        self.stats = self._calculate_stats(features_df)
        self.stats.extraction_time = time.time() - start_time

        return features_df.astype(float).to_numpy()


    def _process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of texts and extract linguistic features."""
        batch_features = []
        for _, row in batch.iterrows():
            text = row["clean_sentence"]
            doc = self.nlp(text)
            features = self._extract_features(doc, text)
            batch_features.append(features)
        return pd.DataFrame(batch_features)

    def _extract_features(self, doc: Doc, text: str) -> Dict[str, float]:
        """Extract various linguistic features from a spaCy Doc."""
        pos_counts = Counter([token.pos_ for token in doc])
        n_tokens = len(doc)
        n_sentences = len(list(doc.sents))

        features = {
            **{f"pos_{pos.lower()}": count / n_tokens for pos, count in pos_counts.items()},
            "n_tokens": n_tokens,
            "n_sentences": n_sentences,
            "avg_sentence_length": n_tokens / n_sentences if n_sentences else 0,
            **self._extract_sentiment_features(text),
            **self._extract_readability_features(text),
        }
        return features

    def _extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features using VADER."""
        scores = self.sia.polarity_scores(text)
        return {f"sentiment_{k}": v for k, v in scores.items()}

    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability features using textstat."""
        try:  # Handle potential errors for short texts
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
                "automated_readability_index": textstat.automated_readability_index(text),
            }
        except Exception:
            return {
                "flesch_reading_ease": np.nan,
                "flesch_kincaid_grade": np.nan,
                "gunning_fog": np.nan,
                "smog_index": np.nan,
                "automated_readability_index": np.nan,
            }



class BusinessFeatureExtractor(BaseFeatureExtractor):
    """Business-specific feature extraction from financial text data."""

    def __init__(self, spacy_model: str = "en_core_web_sm", batch_size: int = 1000):
        super().__init__()
        self.nlp = spacy.load(spacy_model)
        self.batch_size = batch_size
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        self.pattern_categories = {}
        self._initialize_matchers()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BusinessFeatureExtractor":
        """Fit the feature extractor."""
        self._validate_input(X, ["clean_sentence"])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform text data into business-specific features."""

        self._validate_input(X, ["clean_sentence"])
        if not self._is_fitted:
            raise ValueError("BusinessFeatureExtractor must be fitted before transform")

        start_time = time.time()
        all_features = []

        for i in tqdm(range(0, len(X), self.batch_size), desc="Extracting business features"):
            batch = X.iloc[i : i + self.batch_size]
            batch_features = self._process_batch(batch)
            all_features.append(batch_features)

        features_df = pd.concat(all_features, axis=0, ignore_index=True)
        self.stats = self._calculate_stats(features_df)
        self.stats.extraction_time = time.time() - start_time
        return features_df.astype(float).to_numpy()

    def _process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of texts and extract business features."""
        batch_features = []
        for _, row in batch.iterrows():
            text = row["clean_sentence"]
            doc = self.nlp(text)
            features = self._extract_features(doc)
            batch_features.append(features)
        return pd.DataFrame(batch_features)
    

    def _initialize_matchers(self):
        """Initialize PhraseMatcher with business-related terms."""
        business_terms = {
            "financial_metrics": [
                "revenue", "profit", "earnings", "ebitda", "sales", "margin", "growth",
                "market share", "dividend", "cash flow"
            ],
            "business_actions": [
                "merger", "acquisition", "partnership", "investment", "divestment",
                "restructuring", "layoff", "expansion"
            ],
            "market_indicators": [
                "bull", "bear", "volatile", "surge", "plunge", "rally", "correction",
                "uptick", "downturn"
            ],
            "time_references": [
                "quarter", "fiscal", "annual", "yearly", "monthly", "forecast", "outlook",
                "guidance"
            ],
        }

        for category, terms in business_terms.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            self.phrase_matcher.add(category, patterns)
            self.pattern_categories[category] = terms  # Store terms for each category

    def _extract_features(self, doc: spacy.tokens.Doc) -> Dict[str, Union[int, bool]]:
        """Extract business-specific features from a spaCy Doc."""
        money_matches = self.matcher(doc, as_spans=True)
        phrase_matches = self.phrase_matcher(doc)

        features = {
            "has_money_mention": bool(money_matches),
            "n_money_mentions": len(money_matches),
        }

        # Initialize all pattern features to 0
        for category, terms in self.pattern_categories.items():
            for term in terms:
                features[f"{category}_{term}"] = 0

        # Count matches
        for match_id, start, end in phrase_matches:
            category = self.nlp.vocab.strings[match_id]
            matched_text = doc[start:end].text.lower()
            if matched_text in self.pattern_categories[category]:
                features[f"{category}_{matched_text}"] += 1

        return features