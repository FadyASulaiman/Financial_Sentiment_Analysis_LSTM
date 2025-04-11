import numpy as np
from keras.preprocessing.sequence import pad_sequences # type: ignore
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger
from src.utils.constants import SPECIAL_TOKENS

class SequencePadder(PreprocessorBase):
    """Class to pad/truncate sequences to a fixed length"""

    def __init__(self, max_sequence_length=128, padding='post', truncating='post'):
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating
        self.vocab = None
        self.token_to_index = None
        self.index_to_token = None

    def fit(self, X, y=None):
        """Build vocabulary from tokens"""
        logger.info("Building vocabulary from tokens")
        
        # Create a vocabulary of words
        all_tokens = [token for tokens in X for token in tokens]
        unique_tokens = set(all_tokens)
        
        # Add special tokens to vocabulary
        for token in SPECIAL_TOKENS.values():
            unique_tokens.add(token)
        
        self.vocab = sorted(list(unique_tokens))
        self.token_to_index = {token: i + 1 for i, token in enumerate(self.vocab)}  # Reserve 0 for padding
        self.index_to_token = {i + 1: token for i, token in enumerate(self.vocab)}
        self.index_to_token[0] = SPECIAL_TOKENS['PAD']  # Add padding token
        
        logger.info(f"Vocabulary size: {len(self.vocab) + 1}")  # +1 for padding token
        
        return self

    def transform(self, X):
        """Pad/truncate sequences"""
        logger.info(f"Padding/truncating sequences to length {self.max_sequence_length}")

        if self.token_to_index is None:
            self.fit(X)

        # Convert tokens to indices
        sequences = []
        for tokens in X:
            sequence = [self.token_to_index.get(token, self.token_to_index.get(SPECIAL_TOKENS['OOV'], 0)) for token in tokens]
            sequences.append(sequence)

        # Pad/truncate sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding=self.padding,
            truncating=self.truncating,
            value=0  # Padding value
        )

        return padded_sequences