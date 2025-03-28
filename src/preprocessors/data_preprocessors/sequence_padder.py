import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.preprocessing_logger import logger
from utils.constants import SPECIAL_TOKENS

class SequencePadder(PreprocessorBase):
    """Class to pad/truncate sequences to a fixed length"""

    def __init__(self, max_sequence_length=128, padding='post', truncating='post'):
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Pad/truncate sequences"""
        logger.info(f"Padding/truncating sequences to length {self.max_sequence_length}")

        # Create a vocabulary of words
        all_tokens = [token for tokens in X for token in tokens]
        unique_tokens = set(all_tokens)
        token_to_index = {token: i + 1 for i, token in enumerate(unique_tokens)}  # Reserve 0 for padding

        # Convert tokens to indices
        sequences = []
        for tokens in X:
            sequence = [token_to_index.get(token, token_to_index.get(SPECIAL_TOKENS['OOV'], 0)) for token in tokens]
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