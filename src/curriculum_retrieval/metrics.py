"""
Information retrieval evaluation metrics: MRR@10, Recall@K, nDCG@K, Precision@K.
"""

import math
from typing import Dict, List, Optional, Set, Union


def compute_reciprocal_rank(
    ranked_doc_ids: List[str],
    relevant_doc_ids: Set[str],
    k: int = 10,
) -> float:
    """Compute Reciprocal Rank at cutoff K."""
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(
    ranked_doc_ids: List[str],
    relevant_doc_ids: Set[str],
    k: int = 10,
) -> float:
    """Compute Recall at cutoff K."""
    if not relevant_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_doc_ids)
    return min(1.0, hits / float(len(relevant_doc_ids)))


def compute_precision_at_k(
    ranked_doc_ids: List[str],
    relevant_doc_ids: Set[str],
    k: int = 10,
) -> float:
    """Compute Precision at cutoff K."""
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_doc_ids)
    return hits / float(k)


def compute_ndcg_at_k(
    ranked_doc_ids: List[str],
    relevant_doc_ids: Set[str],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at cutoff K."""
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG for binary relevance
    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0.0 else 0.0


def compute_all_metrics(
    ranked_doc_ids: List[str],
    relevant_doc_ids: Union[Set[str], List[str]],
    top_k_pool: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute standard metrics bundle for a single query."""
    rel_set = set(relevant_doc_ids)
    metrics = {
        "mrr@10": compute_reciprocal_rank(ranked_doc_ids, rel_set, k=10),
        "mrr@5": compute_reciprocal_rank(ranked_doc_ids, rel_set, k=5),
        "recall@1": compute_recall_at_k(ranked_doc_ids, rel_set, k=1),
        "recall@5": compute_recall_at_k(ranked_doc_ids, rel_set, k=5),
        "recall@10": compute_recall_at_k(ranked_doc_ids, rel_set, k=10),
        "ndcg@10": compute_ndcg_at_k(ranked_doc_ids, rel_set, k=10),
        "precision@10": compute_precision_at_k(ranked_doc_ids, rel_set, k=10),
    }
    if top_k_pool is not None:
        metrics["candidate_recall@100"] = compute_recall_at_k(top_k_pool, rel_set, k=100)
    return metrics
