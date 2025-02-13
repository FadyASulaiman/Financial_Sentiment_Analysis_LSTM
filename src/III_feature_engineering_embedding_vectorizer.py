from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from transformers import AutoTokenizer, AutoModel
import time
from tqdm import tqdm

from III_feature_engineering_base import BaseFeatureExtractor

class EmbeddingVectorizer(BaseFeatureExtractor):
    """Enhanced word embeddings handler with multiple embedding types support"""
    
    SUPPORTED_FORMATS = {'word2vec', 'glove', 'fasttext'}
    
    def __init__(self,
                 embedding_path: Union[str, Path],
                 embedding_format: str = 'word2vec',
                 embedding_dim: int = 300,
                 aggregation: str = 'mean',
                 cache_dir: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Args:
            embedding_path: Path to embedding file
            embedding_format: Format of embeddings ('word2vec', 'glove', 'fasttext')
            embedding_dim: Dimension of embeddings
            aggregation: Method to aggregate word vectors ('mean', 'max', 'sum')
            cache_dir: Directory to cache processed embeddings
            device: Device to use for computations ('cpu' or 'cuda')
        """
        super().__init__()
        
        if embedding_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Embedding format must be one of {self.SUPPORTED_FORMATS}")
        
        self.embedding_path = Path(embedding_path)
        self.embedding_format = embedding_format
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        
        self.embedding_model = None
        self.vocab = None
        self.embedding_matrix = None
        self.word_index = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EmbeddingVectorizer':
        """Load and prepare embeddings"""
        try:
            self._validate_input(X, ['snippets'])
            
            if self.cache_dir and self._check_cache():
                self._load_from_cache()
            else:
                self._load_embeddings()
                if self.cache_dir:
                    self._save_to_cache()
            
            self._is_fitted = True
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting embedding vectorizer: {str(e)}")
            raise

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform texts to embedding vectors"""
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        try:
            start_time = time.time()
            
            # Process texts in batches
            batch_size = 1000
            n_samples = len(X)
            embeddings = []
            
            for i in tqdm(range(0, n_samples, batch_size), desc="Computing embeddings"):
                batch = X.iloc[i:i+batch_size]
                batch_embeddings = self._texts_to_embeddings(batch['snippets'])
                embeddings.append(batch_embeddings)
            
            features = np.vstack(embeddings)
            
            # Calculate statistics
            self.stats = self._calculate_stats(features)
            self.stats.extraction_time = time.time() - start_time
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error transforming with embeddings: {str(e)}")
            raise

    def _load_embeddings(self):
        """Load embeddings from file"""
        self.logger.info(f"Loading embeddings from {self.embedding_path}")
        
        if self.embedding_format == 'glove':
            word2vec_path = self.embedding_path.with_suffix('.word2vec')
            if not word2vec_path.exists():
                glove2word2vec(str(self.embedding_path), str(word2vec_path))
            self.embedding_model = KeyedVectors.load_word2vec_format(word2vec_path)
        else:
            self.embedding_model = KeyedVectors.load_word2vec_format(self.embedding_path)
        
        self.vocab = set(self.embedding_model.key_to_index.keys())
        self.embedding_matrix = self.embedding_model.vectors
        self.word_index = self.embedding_model.key_to_index

    def _texts_to_embeddings(self, texts: pd.Series) -> np.ndarray:
        """Convert texts to embedding vectors"""
        embeddings = []
        
        for text in texts:
            tokens = text.lower().split()
            word_vectors = []
            
            for token in tokens:
                if token in self.vocab:
                    word_vectors.append(self.embedding_matrix[self.word_index[token]])
            
            if word_vectors:
                if self.aggregation == 'mean':
                    embedding = np.mean(word_vectors, axis=0)
                elif self.aggregation == 'max':
                    embedding = np.max(word_vectors, axis=0)
                else:  # sum
                    embedding = np.sum(word_vectors, axis=0)
            else:
                embedding = np.zeros(self.embedding_dim)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)

    def _check_cache(self) -> bool:
        """Check if cached embeddings exist"""
        if not self.cache_dir:
            return False
        return (self.cache_dir / 'embeddings.npz').exists()

    def _save_to_cache(self):
        """Save processed embeddings to cache"""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / 'embeddings.npz'
        
        np.savez(
            cache_path,
            embedding_matrix=self.embedding_matrix,
            vocab=np.array(list(self.vocab)),
            word_index=np.array(list(self.word_index.items()))
        )

    def _load_from_cache(self):
        """Load embeddings from cache"""
        if not self.cache_dir:
            return
        
        cache_path = self.cache_dir / 'embeddings.npz'
        cached = np.load(cache_path, allow_pickle=True)
        
        self.embedding_matrix = cached['embedding_matrix']
        self.vocab = set(cached['vocab'])
        self.word_index = dict(cached['word_index'])


class TransformerEmbeddings(BaseFeatureExtractor):
    """Enhanced transformer-based embeddings with various pooling strategies"""
    
    SUPPORTED_POOLING = {'cls', 'mean', 'max', 'attention'}
    
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 pooling: str = 'mean',
                 batch_size: int = 32,
                 max_length: int = 128,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 use_attention_mask: bool = True):
        """
        Args:
            model_name: HuggingFace model name
            pooling: Pooling strategy for token embeddings
            batch_size: Processing batch size
            max_length: Maximum sequence length
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
            cache_dir: Directory to cache model and tokenizer
            use_attention_mask: Whether to use attention mask
        """
        super().__init__()
        
        if pooling not in self.SUPPORTED_POOLING:
            raise ValueError(f"Pooling must be one of {self.SUPPORTED_POOLING}")
        
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        self.use_attention_mask = use_attention_mask
        
        self.tokenizer = None
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TransformerEmbeddings':
        """Load and prepare transformer model"""
        try:
            self._validate_input(X, ['snippets'])
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            self._is_fitted = True
            return self
            
        except Exception as e:
            self.logger.error(f"Error loading transformer model: {str(e)}")
            raise

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform texts using transformer embeddings"""
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        try:
            start_time = time.time()
            
            texts = X['snippets'].apply(lambda x: ' '.join(x)).tolist()
            embeddings = []
            
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Computing transformer embeddings"):
                batch_texts = texts[i:i+self.batch_size]
                batch_embeddings = self._process_batch(batch_texts)
                embeddings.append(batch_embeddings)
            
            features = np.vstack(embeddings)
            
            # Calculate statistics
            self.stats = self._calculate_stats(features)
            self.stats.extraction_time = time.time() - start_time
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error transforming with transformer: {str(e)}")
            raise

    def _process_batch(self, texts: List[str]) -> np.ndarray:
        """Process a batch of texts"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if self.pooling == 'cls':
                embeddings = outputs.last_hidden_state[:, 0]
            elif self.pooling == 'mean':
                attention_mask = inputs['attention_mask'] if self.use_attention_mask else None
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            elif self.pooling == 'max':
                embeddings = self._max_pooling(outputs.last_hidden_state)
            else:  # attention
                embeddings = self._attention_pooling(outputs.last_hidden_state, outputs.attentions[-1])
        
        return embeddings.cpu().numpy()

    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean pooling with optional attention mask"""
        if attention_mask is not None:
            token_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
            return token_embeddings.sum(1) / attention_mask.sum(-1, keepdim=True)
        return token_embeddings.mean(dim=1)

    @staticmethod
    def _max_pooling(token_embeddings: torch.Tensor) -> torch.Tensor:
        """Max pooling over token embeddings"""
        return torch.max(token_embeddings, dim=1)[0]

    @staticmethod
    def _attention_pooling(token_embeddings: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        """Attention-weighted pooling"""
        # Average attention weights across all heads
        weights = attention_weights.mean(dim=1)
        # Use weights from CLS token
        weights = weights[:, 0, :].unsqueeze(-1)
        return (token_embeddings * weights).sum(dim=1)

    def get_embedding_dim(self) -> int:
        """Get the dimension of the output embeddings"""
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before getting embedding dimension")
        return self.model.config.hidden_size