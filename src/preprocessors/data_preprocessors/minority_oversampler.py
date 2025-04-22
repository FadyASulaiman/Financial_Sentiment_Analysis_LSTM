import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet
import random
from sklearn.neighbors import NearestNeighbors
import time
from typing import List
from src.utils.loggers.preprocessing_logger import logger


from src.preprocessors.data_preprocessors.base_preprocessor import PreprocessorBase
        
class SentimentBalancer(PreprocessorBase):
    """
    A class to balance sentiment distribution in text datasets using advanced
    synthetic oversampling techniques while preserving additional features.
    """
    
    def __init__(
        self, 
        target_ratio: float = 1.0,
        augmentation_methods: List[str] = ['synonym_replacement', 'random_swap', 'random_insertion'],
        augmentation_prob: float = 0.7,
        random_state: int = 42
    ):
        """Initialize the SentimentBalancer."""
        self.target_ratio = target_ratio
        self.augmentation_methods = augmentation_methods
        self.augmentation_prob = augmentation_prob
        self.random_state = random_state
        self.class_counts_ = None
        self.sentiment_map_ = None
        self.embedding_model = None
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Added flag for tracking additional features
        self.additional_features_ = None
        
        # Import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.SentenceTransformer = SentenceTransformer
            logger.info("Using SentenceTransformer for embeddings")
        except ImportError:
            logger.error("SentenceTransformer Not Available")
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading WordNet for synonym replacement...")
            nltk.download('wordnet')
            
    def set_additional_features(self, additional_features):
        """Set additional features to preserve during balancing."""
        self.additional_features_ = additional_features
        logger.info(f"Set additional features: {list(additional_features.keys())}")
        return self
            
    def fit(self, X, y=None):
        """
        Compute necessary statistics from the dataset to perform balancing.
        """
        start_time = time.time()
        logger.info("Starting fit process...")
        
        # Convert X to list if it's a pandas Series
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()
            
        # Validate that y is provided
        if y is None:
            logger.error("Sentiment labels (y) must be provided to the fit method")
            raise ValueError("Sentiment labels (y) must be provided to the fit method")
            
        # Convert y to list if it's a pandas Series
        if isinstance(y, pd.Series):
            y = y.tolist()
        elif isinstance(y, np.ndarray):
            y = y.tolist()
            
        # Store original data for oversampling
        self.original_X_ = X
        self.original_y_ = y
            
        # Calculate class distribution
        self.class_counts_ = pd.Series(y).value_counts()
        logger.info(f"Class distribution: {dict(self.class_counts_)}")
        
        # Create a mapping of sentiment labels for convenience
        self.sentiment_map_ = {label: idx for idx, label in enumerate(self.class_counts_.index)}
        self.reverse_sentiment_map_ = {idx: label for label, idx in self.sentiment_map_.items()}
        
        # Log if additional features are present
        if self.additional_features_ is not None:
            logger.info(f"Additional features detected: {list(self.additional_features_.keys())}")
        else:
            logger.info("No additional features detected.")

        try:
            logger.info("Loading sentence transformer model")
            self.embedding_model = self.SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer: {str(e)}")
            raise
        
        execution_time = time.time() - start_time
        logger.info(f"Fit completed in {execution_time:.2f} seconds")
        return self
    
    def transform(self, X):
        """
        Transform the input sentences. If needed, balance the dataset by
        oversampling minority classes while preserving additional features.
        """
        start_time = time.time()
        logger.info("Starting transform process...")
        
        # Convert X to list if it's a pandas Series
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()
            
        # Check if fit has been called
        if not hasattr(self, 'class_counts_'):
            logger.error("The fit method must be called before transform")
            raise ValueError("The fit method must be called before transform")
            
        # If the input X is the same as the training data, perform balancing
        if X == self.original_X_:
            logger.info("Input is the same as training data, performing balancing")
            
            # Calculate target count for each class
            max_class_count = self.class_counts_.max()
            result_X = X.copy()
            result_y = self.original_y_.copy()
            
            # Initialize balanced additional features
            self.balanced_additional_features_ = {}
            if self.additional_features_ is not None:
                for feature_name, feature_values in self.additional_features_.items():
                    self.balanced_additional_features_[feature_name] = feature_values.copy()
                    logger.info(f"Initialized balanced feature '{feature_name}' with {len(feature_values)} values")
                    
            # Store original indices for each class
            class_indices = {}
            for idx, label in enumerate(self.original_y_):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
                
            # Generate synthetic samples for each underrepresented class
            for sentiment, count in self.class_counts_.items():
                if count < max_class_count:
                    # Calculate how many samples to generate
                    samples_to_generate = max_class_count - count
                    logger.info(f"Generating {samples_to_generate} synthetic samples for sentiment: {sentiment}")
                    
                    # Get samples of the current sentiment
                    sentiment_indices = class_indices[sentiment]
                    sentiment_samples = [X[i] for i in sentiment_indices]
                    
                    # Generate synthetic samples
                    synthetic_samples = self._generate_synthetic_samples(
                        sentences=sentiment_samples,
                        sentiment=sentiment,
                        num_samples=samples_to_generate
                    )
                    
                    # Add synthetic samples to result
                    result_X.extend(synthetic_samples)
                    result_y.extend([sentiment] * len(synthetic_samples))
                    
                    # Add additional features for synthetic samples
                    if self.additional_features_ is not None:
                        for feature_name, feature_values in self.additional_features_.items():
                            # Randomly sample from the same sentiment class
                            sampled_features = [feature_values[random.choice(sentiment_indices)] for _ in range(len(synthetic_samples))]
                            self.balanced_additional_features_[feature_name].extend(sampled_features)
                            logger.info(f"Added {len(sampled_features)} sampled values for feature '{feature_name}'")
            
            # Create a DataFrame to sort and shuffle the data
            combined_df = pd.DataFrame({
                'X': result_X,
                'y': result_y
            })
            
            # Add additional features to the DataFrame
            if self.additional_features_ is not None:
                for feature_name, feature_values in self.balanced_additional_features_.items():
                    combined_df[feature_name] = feature_values
                    logger.info(f"Added balanced feature '{feature_name}' to combined DataFrame")
                    
            # Verify all features have the correct length
            for col, values in combined_df.items():
                logger.info(f"Column '{col}' has {len(values)} values")
            
            # Shuffle the DataFrame
            combined_df = combined_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            
            # Store the balanced y for potential use by other transformers
            self.balanced_y_ = combined_df['y'].values
            
            # Update balanced additional features after shuffling
            if self.additional_features_ is not None:
                for feature_name in self.balanced_additional_features_.keys():
                    if feature_name in combined_df.columns:
                        self.balanced_additional_features_[feature_name] = combined_df[feature_name].values
                        logger.info(f"Updated balanced feature '{feature_name}' after shuffling")
            
            execution_time = time.time() - start_time
            logger.info(f"Transform completed in {execution_time:.2f} seconds with {len(result_X)} samples")
            
            return combined_df['X'].tolist()
        else:
            # If the input is not the same as training data, just pass it through
            logger.info("Input is different from training data, passing through without changes")
            return X
    
    def get_balanced_y(self):
        """
        Get the balanced sentiment labels after transformation.
        """
        if hasattr(self, 'balanced_y_'):
            return self.balanced_y_
        else:
            logger.warning("balanced_y_ not available, returning original y")
            return self.original_y_
            
    def get_balanced_additional_features(self):
        """Get the balanced additional features after transformation."""
        if hasattr(self, 'balanced_additional_features_'):
            logger.info(f"Returning balanced additional features: {list(self.balanced_additional_features_.keys())}")
            return self.balanced_additional_features_
        else:
            logger.warning("balanced_additional_features_ not available")
            return None
        
    def _generate_synthetic_samples(self, sentences: List[str], sentiment: str, num_samples: int) -> List[str]:
        """
        Generate synthetic text samples using a combination of embedding-based approach
        and text augmentation techniques.
        
        """
        logger.info(f"Generating {num_samples} synthetic samples for {sentiment} sentiment")
        
        if not sentences:
            logger.warning(f"No sentences provided for sentiment {sentiment}")
            return []
            
        synthetic_samples = []
        
        # Generate embeddings for the original sentences
        if self.embedding_model:
            logger.info("Generating embeddings using transformer model")
            embeddings = self.embedding_model.encode(sentences)
        
        # Find nearest neighbors for each sentence
        logger.info("Finding nearest neighbors")
        n_neighbors = min(5 + 1, len(sentences))  # 5 neighbors + the point itself
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(embeddings)
        
        # Generate synthetic samples
        samples_per_method = num_samples // 2  # Half for embedding-based, half for augmentation
        
        # 3.1. Embedding-based generation (interpolation between neighbors)
        logger.info("Generating samples using embedding-based approach")
        embedding_based_samples = []
        
        for i in range(min(samples_per_method, len(sentences))):
            # Select a random sentence
            idx = random.randint(0, len(sentences) - 1)
            base_sentence = sentences[idx]
            
            # Find its neighbors
            distances, indices = nn.kneighbors([embeddings[idx]])
            # Remove the point itself (first neighbor)
            neighbor_indices = indices[0][1:]
            
            if len(neighbor_indices) > 0:
                # Select a random neighbor
                neighbor_idx = random.choice(neighbor_indices)
                neighbor_sentence = sentences[neighbor_idx]
                
                # Generate a new sample by text interpolation
                new_sample = self._interpolate_sentences(base_sentence, neighbor_sentence)
                embedding_based_samples.append(new_sample)
        
        # 3.2. Text augmentation based generation
        logger.info("Generating samples using text augmentation")
        augmentation_samples = []
        
        remaining_samples = num_samples - len(embedding_based_samples)
        
        while len(augmentation_samples) < remaining_samples:
            # Select a random sentence
            base_sentence = random.choice(sentences)
            
            # Apply random augmentation
            method = random.choice(self.augmentation_methods)
            if method == 'synonym_replacement':
                new_sample = self._synonym_replacement(base_sentence)
            elif method == 'random_swap':
                new_sample = self._random_swap(base_sentence)
            elif method == 'random_insertion':
                new_sample = self._random_insertion(base_sentence)
            else:
                new_sample = base_sentence
                
            augmentation_samples.append(new_sample)
        
        # Combine both approaches
        synthetic_samples = embedding_based_samples + augmentation_samples
        
        # Ensure we have exactly num_samples (could be less due to unique neighbor constraints)
        while len(synthetic_samples) < num_samples:
            base_sentence = random.choice(sentences)
            method = random.choice(self.augmentation_methods)
            if method == 'synonym_replacement':
                new_sample = self._synonym_replacement(base_sentence)
            elif method == 'random_swap':
                new_sample = self._random_swap(base_sentence)
            elif method == 'random_insertion':
                new_sample = self._random_insertion(base_sentence)
            else:
                new_sample = base_sentence
                
            synthetic_samples.append(new_sample)
            
        # Ensure we have exactly num_samples (could be more due to batch processing)
        synthetic_samples = synthetic_samples[:num_samples]
        
        return synthetic_samples
    
    def _interpolate_sentences(self, sentence1: str, sentence2: str) -> str:
        """
        Create a new sentence by interpolating between two sentences.
        This is done by taking some words from each sentence.
        
        """
        words1 = sentence1.split()
        words2 = sentence2.split()
        
        # If one of the sentences is empty, return the other
        if not words1:
            return sentence2
        if not words2:
            return sentence1
            
        # Determine the length of the new sentence
        new_length = max(3, (len(words1) + len(words2)) // 2)
        
        # Create a new sentence by randomly selecting words from both sentences
        new_words = []
        
        for i in range(new_length):
            # Randomly choose from which sentence to take the next word
            if random.random() < 0.5 and i < len(words1):
                new_words.append(words1[i])
            elif i < len(words2):
                new_words.append(words2[i])
            elif i < len(words1):
                new_words.append(words1[i])
            else:
                # If we've exhausted both sentences, break the loop
                break
                
        return ' '.join(new_words)
    
    def _synonym_replacement(self, sentence: str, n: int = None) -> str:
        """
        Randomly replace n words in the sentence with one of their synonyms.
        
        """
        words = sentence.split()
        
        # If the sentence has no words, return as is
        if not words:
            return sentence
            
        # Determine the number of words to replace
        if n is None:
            n = max(1, int(len(words) * 0.2))  # Replace ~20% of words
            
        n = min(n, len(words))  # Ensure we don't try to replace more words than exist
        
        # Randomly select words to replace
        random_word_indices = random.sample(range(len(words)), n)
        
        # Replace words with synonyms
        for idx in random_word_indices:
            word = words[idx]
            synonyms = self._get_synonyms(word)
            
            # If synonyms found, replace the word
            if synonyms:
                words[idx] = random.choice(synonyms)
                
        return ' '.join(words)
    
    def _random_swap(self, sentence: str, n: int = None) -> str:
        """
        Randomly swap the positions of n pairs of words in the sentence.

        """
        words = sentence.split()
        
        # If the sentence has less than 2 words, return as is
        if len(words) < 2:
            return sentence
            
        # Determine the number of swaps
        if n is None:
            n = max(1, int(len(words) * 0.1))  # Swap ~10% of word pairs
            
        n = min(n, len(words) // 2)  # Ensure we don't swap more pairs than possible
        
        # Perform random swaps
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    def _random_insertion(self, sentence: str, n: int = None) -> str:
        """
        Randomly insert n synonyms of random words from the sentence into random positions.

        """
        words = sentence.split()
        
        # If the sentence has no words, return as is
        if not words:
            return sentence
            
        # Determine the number of insertions
        if n is None:
            n = max(1, int(len(words) * 0.1))  # Insert ~10% more words
            
        # Perform random insertions
        for _ in range(n):
            # Pick a random word to get its synonym
            word = random.choice(words)
            synonyms = self._get_synonyms(word)
            
            # If synonyms found, insert one at a random position
            if synonyms:
                synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(words))
                words.insert(random_idx, synonym)
                
        return ' '.join(words)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get a list of synonyms for a word using WordNet.
        
        """
        synonyms = []
        
        # Skip very short words and special characters
        if len(word) <= 2 or not word.isalpha():
            return []
            
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
                    
        return synonyms