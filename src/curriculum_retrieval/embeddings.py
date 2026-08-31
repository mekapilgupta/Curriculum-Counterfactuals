"""
Dense embedding interfaces, encoders (multilingual-e5-base, bge-m3, mock),
and disk caching.
"""

import abc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from rich.console import Console
from curriculum_retrieval.provenance import compute_text_hash

console = Console()


class BaseEmbeddingModel(abc.ABC):
    """Abstract base class for multilingual dense embedding models."""

    @abc.abstractmethod
    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """Encode retrieval query strings into normalized float32 vectors."""
        pass

    @abc.abstractmethod
    def encode_passages(self, texts: List[str]) -> np.ndarray:
        """Encode corpus passage/document strings into normalized float32 vectors."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        pass


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic pseudo-embedding encoder for testing and smoke runs."""

    def __init__(self, model_name: str = "mock-encoder", dimension: int = 64):
        self._model_name = model_name
        self.dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _encode_text(self, text: str) -> np.ndarray:
        # Deterministic hashing vector
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(h * (self.dimension // 32 + 1), dtype=np.uint8)[: self.dimension].astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._encode_text(t) for t in texts])

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._encode_text(t) for t in texts])


class SentenceTransformerEncoder(BaseEmbeddingModel):
    """Wrapper around Hugging Face SentenceTransformers for E5 and BGE models."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        normalize: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _lazy_init(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        console.print(f"[bold cyan]Loading SentenceTransformer:[/bold cyan] {self._model_name}")
        self._model = SentenceTransformer(self._model_name, device=self.device)
        self._model.max_seq_length = self.max_length

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        self._lazy_init()
        prefixed = [f"{self.query_prefix}{t}" for t in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        self._lazy_init()
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.array(embeddings, dtype=np.float32)


class EmbeddingCacheManager:
    """Manages embedding vector caching on disk."""

    def __init__(self, cache_dir: str | Path = "data/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, model_name: str, text_hash: str, is_query: bool) -> Path:
        safe_model = model_name.replace("/", "_").replace("\\", "_")
        prefix = "query" if is_query else "passage"
        return self.cache_dir / f"{safe_model}_{prefix}_{text_hash}.npy"

    def get_or_encode_passages(
        self,
        texts: List[str],
        encoder: BaseEmbeddingModel,
    ) -> np.ndarray:
        vectors = []
        missing_indices = []
        missing_texts = []

        for i, text in enumerate(texts):
            thash = compute_text_hash(text)
            cpath = self._get_cache_path(encoder.model_name, thash, is_query=False)
            if cpath.exists():
                vec = np.load(cpath)
                vectors.append((i, vec))
            else:
                missing_indices.append(i)
                missing_texts.append((thash, text))

        if missing_texts:
            encoded = encoder.encode_passages([t for _, t in missing_texts])
            for (idx, (thash, _)), vec in zip(zip(missing_indices, missing_texts), encoded):
                cpath = self._get_cache_path(encoder.model_name, thash, is_query=False)
                np.save(cpath, vec)
                vectors.append((idx, vec))

        vectors.sort(key=lambda x: x[0])
        return np.stack([v for _, v in vectors])

    def get_or_encode_queries(
        self,
        texts: List[str],
        encoder: BaseEmbeddingModel,
    ) -> np.ndarray:
        vectors = []
        missing_indices = []
        missing_texts = []

        for i, text in enumerate(texts):
            thash = compute_text_hash(text)
            cpath = self._get_cache_path(encoder.model_name, thash, is_query=True)
            if cpath.exists():
                vec = np.load(cpath)
                vectors.append((i, vec))
            else:
                missing_indices.append(i)
                missing_texts.append((thash, text))

        if missing_texts:
            encoded = encoder.encode_queries([t for _, t in missing_texts])
            for (idx, (thash, _)), vec in zip(zip(missing_indices, missing_texts), encoded):
                cpath = self._get_cache_path(encoder.model_name, thash, is_query=True)
                np.save(cpath, vec)
                vectors.append((idx, vec))

        vectors.sort(key=lambda x: x[0])
        return np.stack([v for _, v in vectors])
