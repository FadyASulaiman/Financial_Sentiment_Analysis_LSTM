import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.legacy.transformers.base import SequentialFeatureTransformer


class TextCleaner(SequentialFeatureTransformer):
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None

# Now this shit must be refactored, I mean all of it, burned to the ground.
    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        cleaned_text = []
        for text in X:
            # Lowercasing
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            # Tokenization
            tokens = text.split()
            # Remove stopwords
            if self.remove_stopwords:
                tokens = [word for word in tokens if word not in self.stop_words]
            # Lemmatization
            if self.lemmatize:
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            cleaned_text.append(' '.join(tokens))
        return cleaned_text