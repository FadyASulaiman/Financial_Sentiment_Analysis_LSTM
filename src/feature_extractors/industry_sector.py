import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from typing import Dict, List
import pandas as pd
from src.feature_extractors.extractor_base import FeatureExtractorBase
from src.lexicons.financial_lexicon import FinancialLexicon as fl


class IndustrySectorClassifier(FeatureExtractorBase):
    """
    A context-aware feature extractor for classifying financial news sentences into industry sectors.
    """
    
    
    def __init__(self, config: Dict = None):
        """Initialize the IndustrySectorClassifier."""
        
        super().__init__(config)
        
        # Set default configuration values if not provided
        self.input_col = self.config.get('input_col', 'Sentence')
        self.output_col = self.config.get('output_col', 'Sector')
        self.preprocess = self.config.get('preprocess', True)
        self.add_confidence = self.config.get('add_confidence', False)
        self.min_confidence = self.config.get('min_confidence', 0.25)
        self.use_contextual = self.config.get('use_contextual', True)
        self.context_weight = self.config.get('context_weight', 1.5)
        self.allow_multiple = self.config.get('allow_multiple', False)
        self.multi_threshold = self.config.get('multi_threshold', 0.85) # Threshold for multiple sector selection


        self.INDUSTRY_SECTORS = fl.INDUSTRY_SECTORS
        self.INDUSTRY_KEYWORDS = fl.INDUSTRY_KEYWORDS
        self.INDUSTRY_CONTEXT_PATTERNS = fl.INDUSTRY_CONTEXT_PATTERNS
        self.INDUSTRY_OVERLAPS = fl.INDUSTRY_OVERLAPS
        self.INDUSTRY_ENTITY_PATTERNS = fl.INDUSTRY_ENTITY_PATTERNS

        # Add caches for compiled patterns
        self._compiled_CONTEXT_PATTERNS = None

        self._init_nlp()
        
        # Load industry-related stock tickers (optional, could be extended)
        self.industry_tickers = self._load_industry_tickers()
    

    def get_compiled_context_patterns(self):
        """Return compiled context patterns"""
        if self._compiled_CONTEXT_PATTERNS is None:
            self._compiled_CONTEXT_PATTERNS = self._compile_patterns(self.INDUSTRY_CONTEXT_PATTERNS)
        return self._compiled_CONTEXT_PATTERNS


    def _compile_patterns(self, pattern_dict):
        """
        Ensure all regex patterns are compiled

        """
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
            
            # Regular expressions for detecting companies and industries
            self.company_pattern = re.compile(self.INDUSTRY_ENTITY_PATTERNS["company_name"])
            self.ticker_pattern = re.compile(self.INDUSTRY_ENTITY_PATTERNS["ticker_symbol"])
    
    def _load_industry_tickers(self):
        """
        Load mapping of stock tickers to industries (placeholder method).
        In a real implementation, this would load from a database or file.
        
        Returns:
            dict: Mapping of ticker symbols to industry sectors
        """
        # This is a simplified example - in practice, this would be a much larger dataset
        return fl.TICKER_TO_INDUSTRY
    
    def preprocess_text(self, text):
        """
        Preprocess text for classification."""
        if not isinstance(text, str):
            return ""
        
        if not self.preprocess:
            return text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Extract and save company names and tickers for later use
        self.companies = self.company_pattern.findall(text)
        self.tickers = self.ticker_pattern.findall(text)
        
        # Tokenize and lemmatize
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return " ".join(processed_tokens)
        
    def extract_entities(self, text):
        """Extract company names, ticker symbols, and industry terms from text."""
        
        entities = {
            "companies": self.company_pattern.findall(text),
            "tickers": self.ticker_pattern.findall(text),
            "industry_terms": re.findall(re.compile(self.INDUSTRY_ENTITY_PATTERNS["industry_terms"], re.IGNORECASE), text.lower())
        }
        
        # Process tickers to remove '$' prefix
        if entities["tickers"]:
            entities["tickers"] = [ticker[1:] for ticker in entities["tickers"]]  # Remove '$' prefix
            
            # Check if tickers map to known industries
            entities["ticker_industries"] = []
            for ticker in entities["tickers"]:
                if ticker in self.industry_tickers:
                    entities["ticker_industries"].append(
                        (ticker, self.industry_tickers[ticker])
                    )
        
        return entities
    
    def detect_contextual_patterns(self, text):
        """
        Detect contextual patterns in the text related to industries.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of sector matches and their scores
        """
        if not self.use_contextual:
            return {}
            
        scores = {}
        matches = {}

        context_patterns = self.get_compiled_context_patterns()  
        
        # Check contextual patterns for each sector
        for sector in context_patterns:
            sector_score = 0
            sector_matches = []
            
            # Try each pattern
            for pattern, weight in context_patterns[sector]:
                pattern_matches = pattern.findall(text.lower())
                if pattern_matches:
                    sector_score += weight * len(pattern_matches)
                    sector_matches.extend(pattern_matches)
            
            if sector_score > 0:
                scores[sector] = sector_score
                matches[sector] = sector_matches
        
        return {
            "scores": scores,
            "matches": matches
        }
    
    def classify_with_keywords(self, text):
        """
        Classify text using keyword matching.
        
        Args:
            text (str): Text to classify
            
        Returns:
            tuple: (sector_scores, matched_keywords)
        """
        if not isinstance(text, str):
            return {}, {}
            
        text = text.lower()
        scores = {}
        matched_keywords = {}
        
        # Calculate scores for each sector based on keyword matches
        for sector, keywords in self.INDUSTRY_KEYWORDS.items():
            sector_score = 0
            matches = []
            
            for keyword in keywords:
                if keyword.lower() in text:
                    # Weight multi-word keywords higher
                    if " " in keyword:
                        sector_score += 1.5
                    else:
                        sector_score += 1.0
                    matches.append(keyword)
            
            if sector_score > 0:
                # Normalize by the number of keywords
                scores[sector] = sector_score / len(keywords)
                matched_keywords[sector] = matches
        
        return scores, matched_keywords
    
    def check_overlaps(self, scores):
        """
        Check for potential industry overlaps and adjust scores accordingly.
        
        Args:
            scores (dict): Sector scores
            
        Returns:
            dict: Adjusted sector scores
        """
        adjusted_scores = scores.copy()
        
        # Check each defined overlap
        for (sector1, sector2), description in self.INDUSTRY_OVERLAPS.items():
            if sector1 in scores and sector2 in scores:
                # If both sectors have good scores, give a slight boost to both
                if scores[sector1] >= 0.3 and scores[sector2] >= 0.3:
                    boost = min(scores[sector1], scores[sector2]) * 0.2
                    adjusted_scores[sector1] += boost
                    adjusted_scores[sector2] += boost
        
        return adjusted_scores
    
    def classify_text(self, text):
        """
        Classify text into an industry sector using combined keyword and contextual pattern matching.
        
        Args:
            text (str): Text to classify
            
        Returns:
            tuple: (predicted_sector, confidence, all_scores)
        """
        if not isinstance(text, str):
            return "unknown", 0.0, {}
            
        # Preprocess text
        preprocessed = self.preprocess_text(text)
        
        # Extract entities (companies, tickers, industry terms)
        entities = self.extract_entities(text)
        
        # Get scores from keyword matching
        keyword_scores, matched_keywords = self.classify_with_keywords(preprocessed)
        
        # Get scores from contextual pattern matching
        context_results = self.detect_contextual_patterns(text)
        context_scores = context_results.get("scores", {})
        
        # Check ticker-based industry hints
        ticker_boost = {}
        if "ticker_industries" in entities and entities["ticker_industries"]:
            for ticker, industry in entities["ticker_industries"]:
                if industry not in ticker_boost:
                    ticker_boost[industry] = 0
                ticker_boost[industry] += 0.5  # Boost for each ticker matching a known industry
        
        # Combine scores
        combined_scores = {}
        
        # Start with all sectors from all methods
        all_sectors = set(list(keyword_scores.keys()) + list(context_scores.keys()) + list(ticker_boost.keys()))
        
        for sector in all_sectors:
            keyword_score = keyword_scores.get(sector, 0)
            context_score = context_scores.get(sector, 0)
            ticker_score = ticker_boost.get(sector, 0)
            
            # Weight contextual patterns higher
            if self.use_contextual:
                combined_scores[sector] = keyword_score + (context_score * self.context_weight) + ticker_score
            else:
                combined_scores[sector] = keyword_score + ticker_score
        
        # Check for industry overlaps and adjust scores
        combined_scores = self.check_overlaps(combined_scores)
        
        # If no sector matched, return "unknown"
        if not combined_scores:
            return "unknown", 0.0, {}
        
        # Normalize scores
        total_score = sum(combined_scores.values())
        for sector in combined_scores:
            combined_scores[sector] /= total_score
        
        # Get the sector with the highest score
        predicted_sector = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[predicted_sector]
        
        # Apply minimum confidence threshold
        if confidence < self.min_confidence:
            return "unknown", confidence, combined_scores
        
        # Check if we should return multiple sectors
        if self.allow_multiple:
            sectors = []
            for sector, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
                if score >= confidence * self.multi_threshold:
                    sectors.append(sector)
                    # Limit to top 2 sectors
                    if len(sectors) == 2:
                        break
            
            if len(sectors) > 1:
                return "|".join(sectors), confidence, combined_scores
        
        return predicted_sector, confidence, combined_scores
    
    def fit(self, X, y=None):
        """
        No training needed for rule-based approach, but method required for scikit-learn compatibility.
        
        Args:
            X: Input data (ignored)
            y: Target labels (ignored)
            
        Returns:
            self: The classifier instance
        """
        # Nothing to do for rule-based approach
        return self
    
    def transform(self, X):
        """
        Transform the input data by adding sector classification.
        
        Args:
            X (pd.DataFrame or pd.Series or list): Input data containing text samples
            
        Returns:
            pd.DataFrame: Transformed data with sector classification
        """
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
        
        # Get sector predictions
        sector_predictions = []
        confidence_scores = []
        sector_scores = []
        
        for text in X_text:
            sector, confidence, scores = self.classify_text(text)
            sector_predictions.append(sector)
            confidence_scores.append(confidence)
            sector_scores.append(scores)
        
        # Add the predictions as a new column
        result[self.output_col] = sector_predictions
        
        # Optionally add confidence scores
        if self.add_confidence:
            result[f"{self.output_col}_confidence"] = confidence_scores
            
            # Optionally add individual sector scores
            if self.config.get('add_sector_scores', False):
                for sector in self.INDUSTRY_SECTORS:
                    result[f"{sector}_score"] = [scores.get(sector, 0.0) for scores in sector_scores]
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """
        Return the names of features produced by this feature extractor.
        
        Returns:
            List[str]: List of feature names
        """
        feature_names = [self.output_col]
        
        if self.add_confidence:
            feature_names.append(f"{self.output_col}_confidence")
            
            if self.config.get('add_sector_scores', False):
                for sector in self.INDUSTRY_SECTORS:
                    feature_names.append(f"{sector}_score")
                    
        return feature_names
    
    def predict(self, texts):
        """
        Predict the industry sector for input texts.
        
        Args:
            texts (list or str or pd.Series): Input text(s) to classify
            
        Returns:
            list: Predicted industry sectors
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform the texts and extract the predictions
        result = self.transform(texts)
        
        return result[self.output_col].tolist()
    
    def explain_classification(self, text):
        """
        Explain why a text was classified in a particular sector.
        
        Args:
            text (str): Text to explain
            
        Returns:
            dict: Explanation of the classification
        """
        # Preprocess text
        preprocessed = self.preprocess_text(text)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Get scores and matched keywords
        keyword_scores, matched_keywords = self.classify_with_keywords(preprocessed)
        
        # Get context patterns
        context_results = self.detect_contextual_patterns(text)
        context_scores = context_results.get("scores", {})
        context_matches = context_results.get("matches", {})
        
        # Check ticker-based industry hints
        ticker_hints = []
        if "ticker_industries" in entities and entities["ticker_industries"]:
            ticker_hints = entities["ticker_industries"]
        
        # Get final classification
        sector, confidence, all_scores = self.classify_text(text)
        
        # Build explanation
        explanation = {
            "classification": sector,
            "confidence": confidence,
            "all_sector_scores": all_scores,
            "keyword_evidence": matched_keywords,
            "contextual_evidence": context_matches,
            "entities_detected": {
                "companies": entities.get("companies", []),
                "tickers": entities.get("tickers", []),
                "industry_terms": entities.get("industry_terms", [])
            },
            "ticker_industry_hints": ticker_hints
        }
        
        # Check for overlaps
        overlaps = []
        if "|" in sector:
            sectors = sector.split("|")
            for (s1, s2), desc in self.INDUSTRY_OVERLAPS.items():
                if s1 in sectors and s2 in sectors:
                    overlaps.append({"sectors": (s1, s2), "description": desc})
        
        if overlaps:
            explanation["industry_overlaps"] = overlaps
        
        return explanation
    
    def update_keywords(self, sector, new_keywords, replace=False):
        """
        Update the keywords for a specific sector.
        
        Args:
            sector (str): Sector to update
            new_keywords (list): New keywords to add
            replace (bool): Whether to replace existing keywords or append (default: False)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sector not in self.INDUSTRY_SECTORS:
            print(f"Error: Sector '{sector}' not found")
            return False
            
        if replace:
            self.INDUSTRY_KEYWORDS[sector] = new_keywords
        else:
            # Add only unique keywords
            existing_keywords = set(self.INDUSTRY_KEYWORDS[sector])
            for keyword in new_keywords:
                if keyword not in existing_keywords:
                    self.INDUSTRY_KEYWORDS[sector].append(keyword)
        
        return True
    
    def add_context_pattern(self, sector, pattern, weight=1.0):
        """
        Add a new context pattern for a sector.
        
        Args:
            sector (str): Sector to add pattern for
            pattern (str): Regular expression pattern
            weight (float): Weight for the pattern
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sector not in self.INDUSTRY_SECTORS:
            print(f"Error: Sector '{sector}' not found")
            return False
        
        # Compile pattern
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            print(f"Error: Invalid regex pattern '{pattern}'")
            return False
        
        # Initialize pattern list if needed
        if sector not in self.INDUSTRY_CONTEXT_PATTERNS:
            self.INDUSTRY_CONTEXT_PATTERNS[sector] = []
        
        # Add pattern
        self.INDUSTRY_CONTEXT_PATTERNS[sector].append((compiled_pattern, weight))
        
        return True
    
    def add_ticker_industry_mapping(self, ticker, sector):
        """
        Add or update a ticker-to-industry mapping.
        
        Args:
            ticker (str): Stock ticker symbol (without $ prefix)
            sector (str): Industry sector
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sector not in self.INDUSTRY_SECTORS:
            print(f"Error: Sector '{sector}' not found")
            return False
            
        self.industry_tickers[ticker.upper()] = sector
        return True
    
    def analyze_unclassified(self, texts, threshold=0.25):
        """
        Analyze sentences that fall below the confidence threshold.
        
        Args:
            texts (list): List of texts to analyze
            threshold (float): Confidence threshold
            
        Returns:
            dict: Analysis of unclassified texts
        """
        unclassified = []
        low_confidence = []
        sector_distribution = {}
        
        for text in texts:
            sector, confidence, scores = self.classify_text(text)
            
            if confidence < threshold:
                unclassified.append({
                    "text": text,
                    "best_sector": sector,
                    "confidence": confidence,
                    "top_scores": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                })
            elif confidence < 0.5:
                low_confidence.append({
                    "text": text,
                    "sector": sector,
                    "confidence": confidence
                })
            
            # Track sector distribution
            if sector not in sector_distribution:
                sector_distribution[sector] = 0
            sector_distribution[sector] += 1
        
        # Calculate percentage of unclassified
        unclassified_pct = (len(unclassified) / len(texts)) * 100 if texts else 0
        
        return {
            "unclassified_count": len(unclassified),
            "unclassified_percentage": unclassified_pct,
            "low_confidence_count": len(low_confidence),
            "sector_distribution": sector_distribution,
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
        """
        Self-improve by analyzing high-confidence classifications.
        
        Args:
            texts (list): List of texts to analyze
            min_confidence (float): Minimum confidence to consider
            
        Returns:
            dict: Results of self-improvement
        """
        high_confidence_samples = {}
        
        # Collect high confidence samples for each sector
        for text in texts:
            sector, confidence, _ = self.classify_text(text)
            
            if confidence >= min_confidence:
                # Handle multi-sector cases
                if "|" in sector:
                    sectors = sector.split("|")
                    for s in sectors:
                        if s not in high_confidence_samples:
                            high_confidence_samples[s] = []
                        high_confidence_samples[s].append((text, confidence))
                else:
                    if sector not in high_confidence_samples:
                        high_confidence_samples[sector] = []
                    high_confidence_samples[sector].append((text, confidence))
        
        # Use these to improve keyword lists
        improvements = {}
        
        for sector, samples in high_confidence_samples.items():
            if len(samples) < 5:  # Need enough samples
                continue
                
            # Extract potentially useful new keywords
            all_text = " ".join([self.preprocess_text(s[0]) for s in samples])
            words = all_text.split()
            word_counts = Counter(words)
            
            # Find words that might be good indicators but aren't in keywords
            existing_keywords = set([kw.lower() for kw in self.INDUSTRY_KEYWORDS.get(sector, [])])
            
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
                added = self.update_keywords(sector, new_keywords[:5])
                improvements[sector] = new_keywords[:5]
        
        return {
            "sectors_improved": list(improvements.keys()),
            "new_keywords_added": improvements,
            "high_confidence_samples_found": {k: len(v) for k, v in high_confidence_samples.items()}
        }