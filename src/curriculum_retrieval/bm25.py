"""
BM25 lexical index with Hindi and multilingual tokenization.
"""

import re
from typing import Dict, List, Tuple
import numpy as np
from rank_bm25 import BM25Plus


def tokenize_multilingual(text: str) -> List[str]:
    """Tokenize text handling Hindi Devanagari words, English words, and digits."""
    # Match Hindi unicode word sequences or Latin alphanumeric words
    tokens = re.findall(r"[\u0900-\u097F]+|[A-Za-z0-9]+", text.lower())
    return tokens if tokens else text.lower().split()


class BM25Index:
    """BM25 index for sparse lexical retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_ids: List[str] = []
        self.bm25: BM25Plus = None

    def index_documents(self, doc_ids: List[str], texts: List[str]):
        self.corpus_ids = doc_ids
        tokenized_corpus = [tokenize_multilingual(t) for t in texts]
        self.bm25 = BM25Plus(tokenized_corpus, k1=self.k1, b=self.b)

    def query(self, query_text: str, top_k: int = 100) -> List[Tuple[str, float]]:
        tokenized_query = tokenize_multilingual(query_text)
        if not tokenized_query or self.bm25 is None:
            return [(doc_id, 0.0) for doc_id in self.corpus_ids[:top_k]]

        scores = self.bm25.get_scores(tokenized_query)
        # Min-max or standard normalization for downstream fusion
        max_score = float(np.max(scores)) if len(scores) > 0 and np.max(scores) > 0 else 1.0
        norm_scores = scores / max_score

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus_ids[idx], float(norm_scores[idx])) for idx in top_indices]
