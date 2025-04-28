import pandas as pd
from collections import Counter
from src.utils.loggers.data_prep_pipeline_logger import logger

from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase


class VocabularyBuilder(PreprocessorBase):
    """Class responsible for building vocabulary and converting to sequences"""
    
    def __init__(self, min_word_count=2, max_vocab_size=20000):
        """
        Initialize vocabulary builder
        
        Args:
            min_word_count: Minimum frequency to include a word
            max_vocab_size: Maximum vocabulary size
        """
        self.min_word_count = min_word_count
        self.max_vocab_size = max_vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = None
    
    def fit(self, X, y=None):
        """Build vocabulary from lemmatized texts
        
        Args:
            X: pandas.Series where each element is a list of tokens

        """
        logger.info("Building vocabulary")
        
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        
        # Count word frequencies
        all_words = [word for tokens in X for word in tokens]
        self.word_counts = Counter(all_words)
        
        logger.info(f"Total unique words: {len(self.word_counts)}")
        
        # Filter by minimum count
        filtered_words = [word for word, count in self.word_counts.items() 
                         if count >= self.min_word_count]
        
        logger.info(f"Words with count >= {self.min_word_count}: {len(filtered_words)}")
        
        # Sort by frequency (highest first)
        sorted_words = sorted(filtered_words, 
                             key=lambda w: self.word_counts[w], 
                             reverse=True)
        
        # Limit vocabulary size
        if len(sorted_words) > self.max_vocab_size - 2:  # Save space for <PAD> and <UNK>
            sorted_words = sorted_words[:self.max_vocab_size - 2]
        
        # Create word-to-index mapping
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for i, word in enumerate(sorted_words):
            self.word_to_idx[word] = i + 2
        
        # Create index-to-word mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        logger.info(f"Final vocabulary size: {len(self.word_to_idx)}")
        
        # Log most common words
        most_common = self.word_counts.most_common(10)
        logger.info(f"Most common words: {most_common}")
        
        return self
    
    def transform(self, X):
        """Convert lemmatized texts to sequences of word indices
        
        Args:
            X: pandas.Series where each element is a list of tokens
            
        Returns:
            pandas.Series of sequences (each element is a list of indices)
        """
        logger.info("Converting texts to sequences")
        
        if not self.word_to_idx:
            raise ValueError("Vocabulary not built. Call fit() first.")
        
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        
        sequences = X.apply(self._tokens_to_indices)
        
        # Log example
        if not sequences.empty:
            logger.info(f"Example - Words: {X.iloc[0][:10]}")
            logger.info(f"Example - Sequence: {sequences.iloc[0][:10]}")
            
        return sequences
    
    def _tokens_to_indices(self, tokens):
        """Convert a list of tokens to indices"""
        return [self.word_to_idx.get(word, 1) for word in tokens]  # 1 is <UNK>
    
    @property
    def vocabulary(self):
        """Return the vocabulary mapping"""
        return self.word_to_idx
    
    @property
    def vocabulary_size(self):
        """Return the vocabulary size"""
        return len(self.word_to_idx)