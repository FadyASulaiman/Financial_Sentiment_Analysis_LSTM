import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.loggers.eda_logger import logger

class TextProcessor:
    """Class for text processing and feature extraction"""

    def __init__(self, data):
        """
        Initialize with a pandas DataFrame
        
        Args:
            data (pd.DataFrame): Pandas DataFrame with text data
        """

        self.data = data

        # Download necessary NLTK resources if not already available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()



    def clean_text(self, text):
        """Clean and normalize text data"""
        try:
            if not isinstance(text, str):
                return ""

            # lowercase
            text = text.lower()

            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and lemmatize
            cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                             if token not in self.stop_words and len(token) > 2]

            return ' '.join(cleaned_tokens)
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return ""
    
    def process_text(self):
        """Process all text in the dataset"""
        try:
            self.data['cleaned_text'] = self.data['Sentence'].apply(self.clean_text)
            return self.data
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
    
    def get_most_common_words(self, sentiment, top_n=20):
        """Get most common words for a specific sentiment"""
        try:
            # Filter data for the given sentiment
            filtered_data = self.data[self.data['Sentiment'] == sentiment]

            # Join all cleaned text
            all_words = ' '.join(filtered_data['cleaned_text']).split()

            # Count word frequencies
            word_counts = Counter(all_words)

            # Get top N words
            top_words = word_counts.most_common(top_n)

            return top_words
        except Exception as e:
            logger.error(f"Error getting most common words for {sentiment}: {str(e)}")
            return []

    def generate_tfidf_features(self, max_features=1000):
        """Generate TF-IDF features from text"""
        try:
            vectorizer = TfidfVectorizer(max_features=max_features, 
                                         stop_words='english',
                                         ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(self.data['cleaned_text'])

            feature_names = vectorizer.get_feature_names_out()
            return tfidf_matrix, feature_names
        except Exception as e:
            logger.error(f"Error generating TF-IDF features: {str(e)}")
            raise