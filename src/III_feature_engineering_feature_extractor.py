from dataclasses import dataclass
import time
from typing import Dict, List, Set, Optional, Tuple
import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher, PhraseMatcher
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from III_feature_engineering_base import BaseFeatureExtractor
from textstat import textstat
import numpy as np
import pandas as pd
from collections import Counter

@dataclass
class LinguisticStats:
    """Statistics for linguistic features"""
    pos_distribution: Dict[str, int]
    avg_sentence_length: float
    vocab_richness: float
    sentiment_distribution: Dict[str, float]
    complexity_scores: Dict[str, float]

class LinguisticFeatureExtractor(BaseFeatureExtractor):
    """Enhanced linguistic feature extraction"""
    
    FEATURE_GROUPS = {
        'pos': True,
        'syntax': True,
        'sentiment': True,
        'readability': True,
        'lexical': True
    }
    
    def __init__(self,
                 spacy_model: str = 'en_core_web_sm',
                 feature_groups: Optional[Dict[str, bool]] = None,
                 custom_patterns: Optional[List[Dict]] = None,
                 batch_size: int = 1000):
        """
        Args:
            spacy_model: Name of spaCy model to use
            feature_groups: Dictionary of feature groups to extract
            custom_patterns: List of custom patterns to match
            batch_size: Size of batches for processing
        """
        super().__init__()
        
        self.feature_groups = {**self.FEATURE_GROUPS, **(feature_groups or {})}
        self.batch_size = batch_size
        self.custom_patterns = custom_patterns or []
        
        # Initialize NLP components
        self.nlp = spacy.load(spacy_model)
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self._initialize_patterns()
        
        # Track feature names
        self.feature_names: List[str] = []
        self.linguistic_stats: Optional[LinguisticStats] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LinguisticFeatureExtractor':
        """Fit the feature extractor (mainly collects statistics)"""
        self._validate_input(X, ['snippets'])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform texts into linguistic features"""
        if not self._is_fitted:
            raise ValueError("LinguisticFeatureExtractor must be fitted before transform")
        
        try:
            start_time = time.time()
            features = []
            
            # Process in batches
            for i in tqdm(range(0, len(X), self.batch_size), desc="Extracting linguistic features"):
                batch = X.iloc[i:i+self.batch_size]
                batch_features = self._process_batch(batch)
                features.append(batch_features)
            
            # Combine all features
            feature_df = pd.concat(features, axis=0)
            
            # Calculate statistics
            self.stats = self._calculate_stats(feature_df)
            self.stats.extraction_time = time.time() - start_time
            
            # Store feature names
            self.feature_names = feature_df.columns.tolist()
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error extracting linguistic features: {str(e)}")
            raise

    def _process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of texts"""
        features_list = []
        
        for text in batch['snippets']:
            combined_text = ' '.join(text)
            doc = self.nlp(combined_text)
            
            features = {}
            
            if self.feature_groups['pos']:
                features.update(self._extract_pos_features(doc))
            
            if self.feature_groups['syntax']:
                features.update(self._extract_syntax_features(doc))
            
            if self.feature_groups['sentiment']:
                features.update(self._extract_sentiment_features(combined_text))
            
            if self.feature_groups['readability']:
                features.update(self._extract_readability_features(combined_text))
            
            if self.feature_groups['lexical']:
                features.update(self._extract_lexical_features(doc))
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)

    def _extract_pos_features(self, doc: Doc) -> Dict[str, float]:
        """Extract part-of-speech features"""
        pos_counts = Counter(token.pos_ for token in doc)
        total_tokens = len(doc)
        
        features = {
            f'pos_ratio_{pos}': count / total_tokens
            for pos, count in pos_counts.items()
        }
        
        # Add compound features
        features.update({
            'noun_verb_ratio': pos_counts['NOUN'] / max(pos_counts['VERB'], 1),
            'adj_adv_ratio': pos_counts['ADJ'] / max(pos_counts['ADV'], 1),
        })
        
        return features

    def _extract_syntax_features(self, doc: Doc) -> Dict[str, float]:
        """Extract syntactic features"""
        features = {
            'avg_token_length': np.mean([len(token.text) for token in doc]),
            'avg_sentence_length': np.mean([len(sent) for sent in doc.sents]),
            'dependency_distance': np.mean([abs(token.i - token.head.i) for token in doc]),
            'tree_depth': self._get_tree_depth(doc),
            'num_conjunctions': len([token for token in doc if token.dep_ == 'conj']),
        }
        
        return features

    def _extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features"""
        vader_scores = self.sia.polarity_scores(text)
        
        # Add compound sentiment features
        features = {
            f'sentiment_{key}': value
            for key, value in vader_scores.items()
        }
        
        # Add sentiment ratios
        total = vader_scores['pos'] + vader_scores['neg'] + vader_scores['neu']
        features.update({
            'sentiment_ratio_pos': vader_scores['pos'] / total if total > 0 else 0,
            'sentiment_ratio_neg': vader_scores['neg'] / total if total > 0 else 0,
            'sentiment_polarity': vader_scores['pos'] - vader_scores['neg'],
        })
        
        return features

    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability features"""
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'automated_readability_index': textstat.automated_readability_index(text),
        }

    def _extract_lexical_features(self, doc: Doc) -> Dict[str, float]:
        """Extract lexical features"""
        words = [token.text.lower() for token in doc if not token.is_punct]
        unique_words = set(words)
        
        features = {
            'lexical_density': len(unique_words) / max(len(words), 1),
            'avg_word_length': np.mean([len(word) for word in words]),
            'unique_words_ratio': len(unique_words) / max(len(words), 1),
            'hapax_legomena': sum(1 for word in unique_words if words.count(word) == 1) / max(len(words), 1),
        }
        
        return features

    @staticmethod
    def _get_tree_depth(doc: Doc) -> int:
        """Calculate the maximum depth of the dependency tree"""
        def get_depth(token):
            if not list(token.children):
                return 0
            return 1 + max(get_depth(child) for child in token.children)
        
        return max(get_depth(sent.root) for sent in doc.sents)

    def _initialize_patterns(self):
        """Initialize linguistic patterns for matching"""
        # Add custom patterns
        for pattern in self.custom_patterns:
            self.matcher.add(pattern['name'], [pattern['pattern']])


class BusinessFeatureExtractor(BaseFeatureExtractor):
    """Enhanced business-specific feature extraction"""
    
    def __init__(self,
                 spacy_model: str = 'en_core_web_sm',
                 custom_entities: Optional[Dict[str, List[str]]] = None,
                 custom_patterns: Optional[List[Dict]] = None,
                 batch_size: int = 1000):
        """
        Args:
            spacy_model: Name of spaCy model to use
            custom_entities: Dictionary of custom entity lists
            custom_patterns: List of custom patterns to match
            batch_size: Size of batches for processing
        """
        super().__init__()
        
        self.nlp = spacy.load(spacy_model)
        self.batch_size = batch_size
        
        # Initialize business-specific components
        self._initialize_business_components(custom_entities or {})
        self._initialize_patterns(custom_patterns or [])
        
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BusinessFeatureExtractor':
        """Fit the feature extractor"""
        self._validate_input(X, ['snippets'])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform texts into business-specific features"""
        if not self._is_fitted:
            raise ValueError("BusinessFeatureExtractor must be fitted before transform")
        
        try:
            start_time = time.time()
            features = []
            
            # Process in batches
            for i in tqdm(range(0, len(X), self.batch_size), desc="Extracting business features"):
                batch = X.iloc[i:i+self.batch_size]
                batch_features = self._process_batch(batch)
                features.append(batch_features)
            
            # Combine all features
            feature_df = pd.concat(features, axis=0)
            
            # Calculate statistics
            self.stats = self._calculate_stats(feature_df)
            self.stats.extraction_time = time.time() - start_time
            
            # Store feature names
            self.feature_names = feature_df.columns.tolist()
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error extracting business features: {str(e)}")
            raise

    def _initialize_business_components(self, custom_entities: Dict[str, List[str]]):
        """Initialize business-specific components and vocabulary"""
        self.business_vocab = {
            'financial_metrics': [
                'revenue', 'profit', 'earnings', 'ebitda', 'sales',
                'margin', 'growth', 'market share', 'dividend', 'cash flow'
            ],
            'business_actions': [
                'merger', 'acquisition', 'partnership', 'investment',
                'divestment', 'restructuring', 'layoff', 'expansion'
            ],
            'market_indicators': [
                'bull', 'bear', 'volatile', 'surge', 'plunge',
                'rally', 'correction', 'uptick', 'downturn'
            ],
            'time_references': [
                'quarter', 'fiscal', 'annual', 'yearly', 'monthly',
                'forecast', 'outlook', 'guidance'
            ]
        }
        
        # Add custom entities
        for category, terms in custom_entities.items():
            if category in self.business_vocab:
                self.business_vocab[category].extend(terms)
            else:
                self.business_vocab[category] = terms

    def _initialize_patterns(self, custom_patterns: List[Dict]):
        """Initialize pattern matching"""
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        
        # Correct patterns for financial amounts
        money_patterns = [
            # Pattern for numbers followed by magnitude words
            [
                {"LIKE_NUM": True},
                {"LOWER": {"IN": ["million", "billion", "trillion"]}}
            ],
            # Pattern for currency symbols followed by numbers
            [
                {"TEXT": {"IN": ["$", "£", "€", "¥"]}},
                {"LIKE_NUM": True}
            ],
            # Pattern for percentage
            [
                {"LIKE_NUM": True},
                {"TEXT": "%"}
            ],
            # Pattern for currency words followed by numbers
            [
                {"LOWER": {"IN": ["usd", "eur", "gbp", "jpy"]}},
                {"LIKE_NUM": True}
            ],
            # Pattern for numbers followed by currency codes
            [
                {"LIKE_NUM": True},
                {"TEXT": {"IN": ["USD", "EUR", "GBP", "JPY"]}}
            ]
        ]
        
        # Add money patterns to matcher
        self.matcher.add("MONEY_AMOUNT", money_patterns)
        
        # Add phrase patterns for business vocabulary
        for category, terms in self.business_vocab.items():
            patterns = [self.nlp(text.lower()) for text in terms]
            self.phrase_matcher.add(category, patterns)
        
        # Add custom patterns with validation
        for pattern in custom_patterns:
            if self._validate_pattern(pattern):
                self.matcher.add(
                    pattern["name"],
                    [self._convert_pattern(pattern["pattern"])]
                )

    def _validate_pattern(self, pattern: Dict) -> bool:
        """Validate custom pattern format"""
        required_keys = {"name", "pattern"}
        if not all(key in pattern for key in required_keys):
            self.logger.warning(f"Invalid pattern format: {pattern}")
            return False
        return True

    def _convert_pattern(self, pattern: List[Dict]) -> List[Dict]:
        """Convert pattern to spaCy format"""
        converted = []
        for token in pattern:
            token_pattern = {}
            # Handle text matching
            if "TEXT" in token:
                if isinstance(token["TEXT"], str):
                    token_pattern["TEXT"] = token["TEXT"]
                elif isinstance(token["TEXT"], dict) and "IN" in token["TEXT"]:
                    token_pattern["TEXT"] = {"IN": token["TEXT"]["IN"]}
            
            # Handle lemma matching
            if "LEMMA" in token:
                token_pattern["LEMMA"] = token["LEMMA"]
            
            # Handle POS matching
            if "POS" in token:
                token_pattern["POS"] = token["POS"]
            
            # Handle other attributes
            for key in ["LIKE_NUM", "IS_DIGIT", "IS_PUNCT"]:
                if key in token:
                    token_pattern[key] = token[key]
            
            converted.append(token_pattern)
        return converted

    def extract_money_mentions(self, doc: Doc) -> List[Tuple[int, int, str]]:
        """Extract money mentions from text"""
        matches = self.matcher(doc)
        money_mentions = []
        
        for match_id, start, end in matches:
            if doc.vocab.strings[match_id] == "MONEY_AMOUNT":
                span = doc[start:end]
                money_mentions.append((start, end, span.text))
        
        return money_mentions

    def _process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of texts"""
        features_list = []
        
        for text in batch['snippets']:
            combined_text = ' '.join(text)
            doc = self.nlp(combined_text)
            
            features = {}
            features.update(self._extract_entity_features(doc))
            features.update(self._extract_financial_features(doc))
            features.update(self._extract_business_action_features(doc))
            features.update(self._extract_temporal_features(doc))
            features.update(self._extract_market_sentiment_features(doc))
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)

    def _extract_entity_features(self, doc: Doc) -> Dict[str, float]:
        """Extract entity-related features"""
        entities = Counter(ent.label_ for ent in doc.ents)
        
        features = {
            'entity_org_count': entities.get('ORG', 0),
            'entity_money_count': entities.get('MONEY', 0),
            'entity_date_count': entities.get('DATE', 0),
            'entity_percent_count': entities.get('PERCENT', 0),
        }
        
        # Add entity density features
        doc_length = len(doc)
        features.update({
            f'entity_density_{key.lower()}': value / max(doc_length, 1)
            for key, value in entities.items()
        })
        
        return features

    def _extract_financial_features(self, doc: Doc) -> Dict[str, float]:
        """Extract financial-related features"""
        matches = self.matcher(doc)
        financial_matches = [m for m in matches if doc.vocab.strings[m[0]] == 'MONEY_AMOUNT']
        
        features = {
            'has_financial_amount': len(financial_matches) > 0,
            'financial_amount_count': len(financial_matches),
        }
        
        # Add metric mentions
        metric_matches = self.phrase_matcher(doc)
        metric_counts = Counter(doc.vocab.strings[match[0]] for match in metric_matches)
        
        features.update({
            f'metric_{metric.lower()}_count': count
            for metric, count in metric_counts.items()
            if metric in self.business_vocab['financial_metrics']
        })
        
        return features

    def _extract_business_action_features(self, doc: Doc) -> Dict[str, float]:
        """Extract business action-related features"""
        action_matches = self.phrase_matcher(doc)
        action_counts = Counter(doc.vocab.strings[match[0]] for match in action_matches)
        
        features = {
            f'action_{action.lower()}_present': int(action in action_counts)
            for action in self.business_vocab['business_actions']
        }
        
        features.update({
            'total_business_actions': sum(action_counts.values()),
            'unique_business_actions': len(action_counts),
        })
        
        return features

    def _extract_temporal_features(self, doc: Doc) -> Dict[str, float]:
        """Extract temporal references and forecasting indicators"""
        time_matches = sum(1 for token in doc if any(
            term in token.text.lower() for term in self.business_vocab['time_references']
        ))
        
        features = {
            'has_temporal_reference': time_matches > 0,
            'temporal_reference_count': time_matches,
            'future_reference': any(word in doc.text.lower() 
                                  for word in ['will', 'plan', 'expect', 'forecast']),
            'past_reference': any(token.morph.get('Tense') == ['Past'] for token in doc),
        }
        
        return features

    def _extract_market_sentiment_features(self, doc: Doc) -> Dict[str, float]:
        """Extract market sentiment indicators"""
        indicator_matches = self.phrase_matcher(doc)
        indicator_counts = Counter(doc.vocab.strings[match[0]] for match in indicator_matches)
        
        features = {
            f'indicator_{indicator.lower()}_present': int(indicator in indicator_counts)
            for indicator in self.business_vocab['market_indicators']
        }
        
        # Add market sentiment scores
        bullish_terms = set(['bull', 'rally', 'surge', 'uptick', 'growth'])
        bearish_terms = set(['bear', 'plunge', 'downturn', 'decline', 'loss'])
        
        text_lower = doc.text.lower()
        features.update({
            'market_sentiment_score': (
                sum(term in text_lower for term in bullish_terms) -
                sum(term in text_lower for term in bearish_terms)
            ),
            'market_volatility_mention': any(term in text_lower 
                for term in ['volatil', 'fluctuat', 'uncertain'])
        })
        
        return features