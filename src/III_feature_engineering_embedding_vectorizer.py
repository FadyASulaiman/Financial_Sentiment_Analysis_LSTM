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
    """Embedding handling with proper caching"""

    def __init__(self, embedding_path: str, cache_dir: str = None):
        super().__init__()
        self.embedding_path = Path(embedding_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.embeddings = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.cache_dir and (self.cache_dir / "embeddings.kv").exists():
            self.embeddings = KeyedVectors.load(str(self.cache_dir / "embeddings.kv"))
        else:
            if self.embedding_path.suffix == ".txt":
                glove2word2vec(str(self.embedding_path), str(self.embedding_path.with_suffix(".kv")))
                self.embeddings = KeyedVectors.load_word2vec_format(str(self.embedding_path.with_suffix(".kv")))
            else:
                self.embeddings = KeyedVectors.load_word2vec_format(str(self.embedding_path))
            
            if self.cache_dir:
                self.embeddings.save(str(self.cache_dir / "embeddings.kv"))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        embeddings = []
        for text in X["clean_sentence"]:
            vectors = [self.embeddings[word] for word in text.split() if word in self.embeddings]
            embeddings.append(np.mean(vectors, axis=0) if vectors else np.zeros(300))
        return np.array(embeddings)
