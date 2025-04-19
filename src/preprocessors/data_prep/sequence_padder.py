from keras.preprocessing.sequence import pad_sequences # type: ignore
import pandas as pd # type: ignore
from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
from src.utils.loggers.preprocessing_logger import logger


class SequencePadder(PreprocessorBase):
    """Class responsible for sequence padding"""
    
    def __init__(self, max_sequence_length=128, padding='post', truncating='post'):
        """
        Initialize sequence padder

        """
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Pad sequences to fixed length
        
        Args:
            X: pandas.Series where each element is a list of indices
            
        Returns:
            numpy.ndarray of padded sequences
        """
        logger.info(f"Padding sequences to length {self.max_sequence_length}")
        
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        
        # Convert Series of lists to list of lists for pad_sequences
        sequences_list = X.tolist()
        
        padded_sequences = pad_sequences(
            sequences_list, 
            maxlen=self.max_sequence_length,
            padding=self.padding,
            truncating=self.truncating,
            value=0  # 0 is <PAD>
        )
        
        logger.info(f"Padded sequences shape: {padded_sequences.shape}")
        
        return padded_sequences