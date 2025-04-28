import tensorflow as tf

from src.legacy.transformers.base import SequentialFeatureTransformer


class TokenizerTransformer(SequentialFeatureTransformer):
    def __init__(self, num_words=20000, oov_token='<OOV>', max_length=128, padding='post', truncating='post'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.max_length = max_length
        self.padding = padding
        self.truncating = truncating
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.num_words, oov_token=self.oov_token)
    
    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X)
        return self
    
    def transform(self, X):
        sequences = self.tokenizer.texts_to_sequences(X)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_length, padding=self.padding, truncating=self.truncating)
        return padded
    
if __name__ == '__main__':
    # To run: $ python -m src.transformers.tokenizer_transformer
    # Example usage and simple test.
    texts = [
        "Hello world, how are you?",
        "I am fine, thank you.",
        "Transformers are very useful for NLP tasks."
    ]
    
    # Create an instance of the transformer.
    tt = TokenizerTransformer(num_words=100, max_length=10)
    
    # Fit and transform the texts.
    padded_sequences = tt.fit_transform(texts)
    
    print("Padded sequences:")
    print(padded_sequences)