import logging
from pathlib import Path
import time
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from III_feature_engineering_base import BaseFeatureExtractor


class EmbeddingVectorizer(BaseFeatureExtractor):
    """Word embedding vectorizer using pre-trained models."""

    SUPPORTED_FORMATS = {"word2vec", "glove", "fasttext"}

    def __init__(
        self,
        embedding_path: Union[str, Path],
        embedding_format: str = "word2vec",
        aggregation: str = "mean",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        if embedding_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported embedding format: {embedding_format}. "
                f"Choose from {self.SUPPORTED_FORMATS}"
            )

        self.embedding_path = Path(embedding_path)
        self.embedding_format = embedding_format
        self.aggregation = aggregation
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.embeddings: Optional[KeyedVectors] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "EmbeddingVectorizer":
        """Load and prepare the embedding model."""
        self._validate_input(X, ["clean_sentence"])

        cache_path = self._get_cache_path()
        if self.cache_dir and cache_path.exists():
            self._load_from_cache(cache_path)
        else:
            self._load_embeddings()
            if self.cache_dir:
                self._save_to_cache(cache_path)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform text data into embedding vectors."""
        self._validate_input(X, ["clean_sentence"])
        if not self.embeddings:
            raise ValueError("Embeddings must be loaded before transform")

        start_time = time.time()
        embeddings = []
        for text in tqdm(X["clean_sentence"], desc="Generating embeddings"):
            embeddings.append(self._text_to_embedding(text))
        features = np.array(embeddings)

        self.stats = self._calculate_stats(features)
        self.stats.extraction_time = time.time() - start_time
        return features

    def _load_embeddings(self):
        """Load embeddings from file based on format."""
        if self.embedding_format == "glove":
            word2vec_path = self.embedding_path.with_suffix(".word2vec")
            if not word2vec_path.exists():
                glove2word2vec(str(self.embedding_path), str(word2vec_path))
            self.embeddings = KeyedVectors.load_word2vec_format(word2vec_path)
        else:  # word2vec or fasttext
            self.embeddings = KeyedVectors.load_word2vec_format(self.embedding_path)

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for a single text."""
        tokens = text.split()
        word_vectors = [self.embeddings[token] for token in tokens if token in self.embeddings]

        if word_vectors:
            if self.aggregation == "mean":
                return np.mean(word_vectors, axis=0)
            elif self.aggregation == "max":
                return np.max(word_vectors, axis=0)
            elif self.aggregation == "sum":
                return np.sum(word_vectors, axis=0)
            else:
                raise ValueError(f"Invalid aggregation method: {self.aggregation}")
        else:
            return np.zeros(self.embeddings.vector_size)

    def _get_cache_path(self) -> Path:
        """Generate cache file path."""
        filename = f"{self.embedding_path.stem}_{self.aggregation}.npz"
        return self.cache_dir / filename

    def _save_to_cache(self, cache_path: Path):
        """Save embeddings to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=self.embeddings.vectors)

    def _load_from_cache(self, cache_path: Path):
        """Load embeddings from cache."""
        loaded = np.load(cache_path)
        self.embeddings = KeyedVectors(vector_size=loaded["embeddings"].shape[1])
        self.embeddings.add_vectors(
            keys=np.array(list(self.embeddings.key_to_index)), weights=loaded["embeddings"]
        )


class TransformerEmbeddings(BaseFeatureExtractor):
    """Transformer-based embedding vectorizer using Hugging Face models."""

    SUPPORTED_POOLING = {"cls", "mean", "max"}

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling: str = "mean",
        batch_size: int = 32,
        max_length: int = 128,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        if pooling not in self.SUPPORTED_POOLING:
            raise ValueError(f"Invalid pooling method: {pooling}. Choose from {self.SUPPORTED_POOLING}")

        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TransformerEmbeddings":
        """Load and prepare the transformer model and tokenizer."""
        self._validate_input(X, ["clean_sentence"])

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device)
        self.model.eval()
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform text data into transformer embeddings."""

        self._validate_input(X, ["clean_sentence"])
        if not self.tokenizer or not self.model:
            raise ValueError("Model and tokenizer must be loaded before transform")

        start_time = time.time()
        all_embeddings = []
        for i in tqdm(range(0, len(X), self.batch_size), desc="Generating embeddings"):
            batch = X.iloc[i : i + self.batch_size]
            batch_embeddings = self._process_batch(batch["clean_sentence"].tolist())
            all_embeddings.append(batch_embeddings)

        features = np.vstack(all_embeddings)
        self.stats = self._calculate_stats(features)
        self.stats.extraction_time = time.time() - start_time
        return features

    def _process_batch(self, texts: List[str]) -> np.ndarray:
        """Process a batch of texts and generate embeddings."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._pool_embeddings(outputs.last_hidden_state, inputs["attention_mask"])

        return embeddings.cpu().numpy()

    def _pool_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings based on the chosen strategy."""
        if self.pooling == "cls":
            return embeddings[:, 0]
        elif self.pooling == "mean":
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            return masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        elif self.pooling == "max":
            return torch.max(embeddings, dim=1)[0]
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling}")