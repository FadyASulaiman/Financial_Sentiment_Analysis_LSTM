from collections import Counter
import re
import pandas as pd
from typing import Dict, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.lexicons.financial_lexicon import FinancialLexicon as fl


class FinancialEventClassifier(FeatureExtractorBase):
    """
    A context-aware feature extractor for classifying financial news sentences into predefined event categories.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the FinancialEventClassifier with context-aware pattern matching.
        
        """

        # Store raw patterns instead of compiled ones
        self.raw_CONTEXT_PATTERNS = fl.CONTEXT_PATTERNS
        self.raw_NEGATIVE_CONTEXTS = fl.NEGATIVE_CONTEXTS
        
        # Add caches for compiled patterns
        self._compiled_CONTEXT_PATTERNS = None
        self._compiled_NEGATIVE_CONTEXTS = None

        self.EVENT_CATEGORIES = fl.EVENT_CATEGORIES
        self.EVENT_KEYWORDS =fl.EVENT_KEYWORDS
        self.ENTITY_PATTERNS = fl.ENTITY_PATTERNS
        self.FINANCIAL_VERBS = fl.FINANCIAL_VERBS


        super().__init__(config)
        
        # Set default configuration values if not provided
        self.input_col = self.config.get('input_col', 'Sentence')
        self.output_col = self.config.get('output_col', 'Event')
        self.preprocess = self.config.get('preprocess', True)
        self.add_confidence = self.config.get('add_confidence', False)
        self.min_confidence = self.config.get('min_confidence', 0.2)
        self.use_contextual = self.config.get('use_contextual', True)
        self.context_weight = self.config.get('context_weight', 1.4) # Weight for contextual pattern match scores
        
        self._init_nlp()


    def get_compiled_context_patterns(self):
        """Return compiled context patterns"""
        if self._compiled_CONTEXT_PATTERNS is None:
            self._compiled_CONTEXT_PATTERNS = self._compile_patterns(self.raw_CONTEXT_PATTERNS)
        return self._compiled_CONTEXT_PATTERNS

    def get_compiled_negative_contexts(self):
        """Return compiled negative context patterns"""
        if self._compiled_NEGATIVE_CONTEXTS is None:
            self._compiled_NEGATIVE_CONTEXTS = self._compile_patterns(self.raw_NEGATIVE_CONTEXTS)
        return self._compiled_NEGATIVE_CONTEXTS
        
    def _compile_patterns(self, pattern_dict):
        """Ensure all regex patterns are compiled"""
        compiled_dict = {}
        for category, patterns in pattern_dict.items():
            compiled_patterns = []
            for pattern_tuple in patterns:
                pattern, weight = pattern_tuple
                # Compile if it's a string
                if isinstance(pattern, str):
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    compiled_patterns.append((compiled_pattern, weight))
                else:
                    # It's already a compiled pattern
                    compiled_patterns.append(pattern_tuple)
            compiled_dict[category] = compiled_patterns
        return compiled_dict
    

    def _init_nlp(self):
        """Initialize NLP components"""
        if self.preprocess:
            # Download necessary NLTK resources
            try:
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('stopwords')
                nltk.download('wordnet')
                nltk.download('punkt')
                
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Regular expressions for detecting financial patterns
            self.money_pattern = re.compile(r'(\$|€|£|USD|EUR|GBP|million|billion|mn|bn|\d+\.\d+|\d+%)')
            self.ticker_pattern = re.compile(r'\$[A-Z]{1,5}')
            
            # Add named entity recognition patterns
            self.companies_pattern = re.compile(r'([A-Z][a-z]*\.?\s)?([A-Z][a-z]+\s)*[A-Z][a-z]*\.?(\s(Inc|Corp|Co|Ltd|LLC|Group|SA|AG|SE|NV|PLC|GmbH)\.?)?')
    
    def preprocess_text(self, text):
        """Preprocess text for classification with more advanced techniques."""
        if not isinstance(text, str):
            return ""
        
        if not self.preprocess:
            return text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Replace ticker symbols with a token
        text = self.ticker_pattern.sub(" TICKERSYMBOL ", text)
        
        # Replace monetary values with a token but preserve the value
        text = self.money_pattern.sub(lambda m: f" MONEYVALUE_{m.group(0).replace(' ', '_')} ", text)
        
        # Identify company names
        text = self.companies_pattern.sub(" COMPANY ", text)
        
        # Tokenize and lemmatize
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return " ".join(tokens)
        
    def find_entities(self, text):
        """Extract financial entities from text."""
        entities = {}
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            pattern = re.compile(pattern, re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = matches
                
        return entities
    
    def extract_sentence_structure(self, text):
        """Extract basic sentence structure features."""
        # Tokenize
        tokens = nltk.word_tokenize(text.lower())
        
        # Get parts of speech with a basic approach
        nouns = []
        verbs = []
        
        for token in tokens:
            if token.isalpha() and len(token) > 2:
                # This is a very basic approximation - ideally use proper POS tagging
                if token.endswith('ing') or token.endswith('ed') or token.endswith('s'):
                    verbs.append(token)
                else:
                    nouns.append(token)
        
        # Identify verb sentiment
        verb_sentiment = "neutral"
        for verb in verbs:
            verb_stem = self.lemmatizer.lemmatize(verb, 'v')
            if verb_stem in self.FINANCIAL_VERBS["positive"]:
                verb_sentiment = "positive"
                break
            elif verb_stem in self.FINANCIAL_VERBS["negative"]:
                verb_sentiment = "negative"
                break
        
        return {
            "tokens": tokens,
            "nouns": nouns,
            "verbs": verbs,
            "verb_sentiment": verb_sentiment,
            "token_count": len(tokens)
        }
    def detect_contextual_patterns(self, text):
        """Detect contextual patterns in the text."""
        if not self.use_contextual:
            return {}
            
        scores = {}
        matches = {}

        # Use getters instead of compiling every time
        context_patterns = self.get_compiled_context_patterns()  
        negative_contexts = self.get_compiled_negative_contexts()
        
        # Check contextual patterns for each category
        for category in context_patterns:
            category_score = 0
            category_matches = []
            
            # Try each pattern
            for pattern, weight in context_patterns[category]:
                pattern_matches = pattern.findall(text.lower())
                if pattern_matches:
                    category_score += weight * len(pattern_matches)
                    category_matches.extend(pattern_matches)
            
            if category_score > 0:
                scores[category] = category_score
                matches[category] = category_matches
                
        # Check negative contexts
        for category in negative_contexts:
            if category in scores:
                for pattern, weight in negative_contexts[category]:
                    neg_matches = pattern.findall(text.lower())
                    if neg_matches:
                        # Reduce or negate the score based on negative context
                        scores[category] -= weight * len(neg_matches)
                        
                # Remove if score is negative after accounting for negative context
                if scores[category] <= 0:
                    del scores[category]
                    del matches[category]
        
        return {
            "scores": scores,
            "matches": matches
        }
    
    def classify_with_keywords(self, text):
        """Classify text using keyword matching from the original approach."""
        if not isinstance(text, str):
            return {}, {}
            
        text = text.lower()
        scores = {}
        matched_keywords = {}
        
        # Calculate scores for each category based on keyword matches
        for category, keywords in self.EVENT_KEYWORDS.items():
            category_score = 0
            matches = []
            
            for keyword in keywords:
                if keyword.lower() in text:
                    # Weight multi-word keywords higher
                    if " " in keyword:
                        category_score += 1.5
                    else:
                        category_score += 1.0
                    matches.append(keyword)
            
            if category_score > 0:
                # Normalize by the number of keywords
                scores[category] = category_score / len(keywords)
                matched_keywords[category] = matches
        
        return scores, matched_keywords
    
    def classify_text(self, text):
        """Classify text using combined keyword and contextual pattern matching."""
        if not isinstance(text, str):
            return "other", 1.0, {"other": 1.0}
            
        # Preprocess text
        preprocessed = text.lower()
        
        # Get scores from keyword matching
        keyword_scores, matched_keywords = self.classify_with_keywords(preprocessed)
        
        # Get scores from contextual pattern matching
        context_results = self.detect_contextual_patterns(text)
        context_scores = context_results.get("scores", {})
        
        # Extract entities and sentence structure for additional features
        entities = self.find_entities(text)
        sentence_features = self.extract_sentence_structure(text)
        
        # Combine scores
        combined_scores = {}
        
        # Add all categories from both methods
        all_categories = set(list(keyword_scores.keys()) + list(context_scores.keys()))
        
        for category in all_categories:
            keyword_score = keyword_scores.get(category, 0)
            context_score = context_scores.get(category, 0)
            
            # Weight contextual patterns higher
            if self.use_contextual:
                combined_scores[category] = keyword_score + (context_score * self.context_weight)
            else:
                combined_scores[category] = keyword_score
        
        # Apply additional heuristics
        
        # 1. If we have money entities and financial verbs, boost certain categories
        if 'money' in entities and sentence_features['verbs']:
            if sentence_features['verb_sentiment'] == 'positive':
                # Boost positive financial events
                for category in ['earnings', 'investment', 'expansion']:
                    if category in combined_scores:
                        combined_scores[category] *= 1.3
            elif sentence_features['verb_sentiment'] == 'negative':
                # Boost negative financial events
                for category in ['layoff', 'restructuring', 'debt_financing']:
                    if category in combined_scores:
                        combined_scores[category] *= 1.3
        
        # 2. For very short messages, require higher confidence
        if sentence_features['token_count'] < 8:
            for category in combined_scores:
                combined_scores[category] *= 0.8
        
        # If no category matched, return "other"
        if not combined_scores:
            return "other", 1.0, {"other": 1.0}
        
        # Normalize scores
        total_score = sum(combined_scores.values())
        for category in combined_scores:
            combined_scores[category] /= total_score
        
        # Get the category with the highest score
        predicted_category = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[predicted_category]
        
        # Apply minimum confidence threshold
        if confidence < self.min_confidence:
            return "other", confidence, combined_scores
        
        return predicted_category, confidence, combined_scores
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transform the input data by adding event classification."""
        # Copy the input data to avoid modifying the original
        if isinstance(X, pd.DataFrame):
            result = X.copy()
            X_text = X[self.input_col]
        else:
            # If X is a Series or list, convert to DataFrame
            if isinstance(X, pd.Series):
                X_text = X
                result = pd.DataFrame({self.input_col: X})
            else:
                X_text = X
                result = pd.DataFrame({self.input_col: X})
        
        # Convert to list if Series
        if isinstance(X_text, pd.Series):
            X_text = X_text.tolist()
        
        # Get event predictions
        event_predictions = []
        confidence_scores = []
        category_scores = []
        
        for text in X_text:
            category, confidence, scores = self.classify_text(text)
            event_predictions.append(category)
            confidence_scores.append(confidence)
            category_scores.append(scores)
        
        # Add the predictions as a new column
        result[self.output_col] = event_predictions
        
        # Optionally add confidence scores
        if self.add_confidence:
            result[f"{self.output_col}_confidence"] = confidence_scores
            
            # Optionally add individual category scores
            if self.config.get('add_category_scores', False):
                for category in self.EVENT_CATEGORIES:
                    result[f"{category}_score"] = [scores.get(category, 0.0) for scores in category_scores]
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Return the names of features produced by this feature extractor."""
        feature_names = [self.output_col]
        
        if self.add_confidence:
            feature_names.append(f"{self.output_col}_confidence")
            
            if self.config.get('add_category_scores', False):
                for category in self.EVENT_CATEGORIES:
                    feature_names.append(f"{category}_score")
                    
        return feature_names
    
    def predict(self, texts):
        """Predict the event category for input texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform the texts and extract the predictions
        result = self.transform(texts)
        
        return result[self.output_col].tolist()
    
    def explain_classification(self, text):
        """Explain why a text was classified in a particular way."""
        # Preprocess text
        preprocessed = text.lower()
        
        # Get scores and matched keywords
        keyword_scores, matched_keywords = self.classify_with_keywords(preprocessed)
        
        # Get context patterns
        context_results = self.detect_contextual_patterns(text)
        context_scores = context_results.get("scores", {})
        context_matches = context_results.get("matches", {})
        
        # Get entities
        entities = self.find_entities(text)
        
        # Get sentence structure
        sentence_features = self.extract_sentence_structure(text)
        
        # Get final classification
        category, confidence, all_scores = self.classify_text(text)
        
        # Build explanation
        explanation = {
            "classification": category,
            "confidence": confidence,
            "all_category_scores": all_scores,
            "keyword_evidence": matched_keywords,
            "contextual_evidence": context_matches,
            "entities_detected": entities,
            "sentence_features": {
                "verbs": sentence_features["verbs"],
                "verb_sentiment": sentence_features["verb_sentiment"],
                "length": sentence_features["token_count"]
            }
        }
        
        return explanation
    
    def update_keywords(self, category, new_keywords, replace=False):
        """Update the keywords for a specific category."""
        if category not in self.EVENT_CATEGORIES:
            print(f"Error: Category '{category}' not found")
            return False
            
        if replace:
            self.EVENT_KEYWORDS[category] = new_keywords
        else:
            # Add only unique keywords
            existing_keywords = set(self.EVENT_KEYWORDS[category])
            for keyword in new_keywords:
                if keyword not in existing_keywords:
                    self.EVENT_KEYWORDS[category].append(keyword)
        
        return True
    
    def add_context_pattern(self, category, pattern, weight=1.0):
        """Add a new context pattern for a category."""
        if category not in self.EVENT_CATEGORIES:
            print(f"Error: Category '{category}' not found")
            return False
        
        # Compile pattern
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            print(f"Error: Invalid regex pattern '{pattern}'")
            return False
        
        # Initialize pattern list if needed
        if category not in self.CONTEXT_PATTERNS:
            self.CONTEXT_PATTERNS[category] = []
        
        # Add pattern
        self.CONTEXT_PATTERNS[category].append((compiled_pattern, weight))
        
        return True
    
    def analyze_unclassified(self, texts, threshold=0.3):
        """Analyze sentences that fall below the confidence threshold."""
        unclassified = []
        low_confidence = []
        category_distribution = {}
        
        for text in texts:
            category, confidence, scores = self.classify_text(text)
            
            if confidence < threshold:
                unclassified.append({
                    "text": text,
                    "best_category": category,
                    "confidence": confidence,
                    "top_scores": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                })
            elif confidence < 0.5:
                low_confidence.append({
                    "text": text,
                    "category": category,
                    "confidence": confidence
                })
            
            # Track category distribution
            if category not in category_distribution:
                category_distribution[category] = 0
            category_distribution[category] += 1
        
        # Calculate percentage of unclassified
        unclassified_pct = (len(unclassified) / len(texts)) * 100 if texts else 0
        
        return {
            "unclassified_count": len(unclassified),
            "unclassified_percentage": unclassified_pct,
            "low_confidence_count": len(low_confidence),
            "category_distribution": category_distribution,
            "unclassified_examples": unclassified[:10],  # First 10 examples
            "common_patterns": self._extract_common_patterns(unclassified)
        }
    
    def _extract_common_patterns(self, unclassified_texts):
        """Extract common patterns from unclassified texts"""
        if not unclassified_texts:
            return []
            
        # Extract words that appear frequently
        all_words = []
        for item in unclassified_texts:
            tokens = nltk.word_tokenize(item["text"].lower())
            all_words.extend([t for t in tokens if t.isalpha() and t not in self.stop_words])
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Get most common words
        common_words = word_counts.most_common(20)
        
        # Look for bigrams
        bigrams = []
        for item in unclassified_texts:
            tokens = nltk.word_tokenize(item["text"].lower())
            for i in range(len(tokens) - 1):
                if tokens[i].isalpha() and tokens[i+1].isalpha():
                    bigrams.append((tokens[i], tokens[i+1]))
        
        # Count bigram frequencies
        bigram_counts = Counter(bigrams)
        
        # Get most common bigrams
        common_bigrams = bigram_counts.most_common(10)
        
        return {
            "common_words": common_words,
            "common_bigrams": common_bigrams
        }
    
    def self_improve(self, texts, min_confidence=0.7):
        """Self-improve by analyzing high-confidence classifications."""
        high_confidence_samples = {}
        
        # Collect high confidence samples for each category
        for text in texts:
            category, confidence, _ = self.classify_text(text)
            
            if confidence >= min_confidence:
                if category not in high_confidence_samples:
                    high_confidence_samples[category] = []
                    
                high_confidence_samples[category].append((text, confidence))
        
        # Use these to improve keyword lists
        improvements = {}
        
        for category, samples in high_confidence_samples.items():
            if len(samples) < 5:  # Need enough samples
                continue
                
            # Extract potentially useful new keywords
            all_text = " ".join([self.preprocess_text(s[0]) for s in samples])
            words = all_text.split()
            word_counts = Counter(words)
            
            # Find words that might be good indicators but aren't in keywords
            existing_keywords = set([kw.lower() for kw in self.EVENT_KEYWORDS.get(category, [])])
            
            new_keywords = []
            for word, count in word_counts.most_common(50):
                # Filter to keep only good potential keywords
                if (len(word) > 3 and 
                    word not in self.stop_words and 
                    word.lower() not in existing_keywords and
                    count >= 3):  # Appears in multiple samples
                    new_keywords.append(word)
            
            # Add the best new keywords
            if new_keywords:
                added = self.update_keywords(category, new_keywords[:5])
                improvements[category] = new_keywords[:5]
        
        return {
            "categories_improved": list(improvements.keys()),
            "new_keywords_added": improvements,
            "high_confidence_samples_found": {k: len(v) for k, v in high_confidence_samples.items()}
        }