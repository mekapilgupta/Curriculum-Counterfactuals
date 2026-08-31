"""
Core retrieval engine implementing systems R0 through R6.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from rich.console import Console
from curriculum_retrieval.bm25 import BM25Index
from curriculum_retrieval.concepts import ConceptManager
from curriculum_retrieval.embeddings import BaseEmbeddingModel, EmbeddingCacheManager
from curriculum_retrieval.explainability import create_explainability_trace
from curriculum_retrieval.graph_index import ConceptGraphIndex
from curriculum_retrieval.schemas import (
    ConceptRecord,
    ExplainabilityTrace,
    QueryConceptRecord,
    QueryRecord,
    SourceDocumentRecord,
    TranslationRecord,
)

console = Console()


def parse_grade_to_int(grade_str: str) -> int:
    """Parse grade string into an integer or ordinal index for distance calculation."""
    if not grade_str:
        return 6
    match = re.search(r"\d+", str(grade_str))
    if match:
        return int(match.group())
    lower = str(grade_str).lower()
    if "kindergarten" in lower or "k" in lower:
        return 0
    if "elementary" in lower:
        return 3
    if "middle" in lower:
        return 7
    if "high" in lower:
        return 10
    return 6


class RetrievalEngine:
    """
    Executes R0 (dense raw), R1 (BM25), R2 (hybrid), R3 (grade-aware),
    R4 (metadata-aware), R5 (concept fusion), and R6 (concept candidate generation).
    """

    def __init__(
        self,
        documents: List[SourceDocumentRecord],
        doc_translations: Dict[str, TranslationRecord],
        doc_concepts: Dict[str, ConceptRecord],
        encoder: BaseEmbeddingModel,
        embedding_cache: EmbeddingCacheManager,
        bm25_index: Optional[BM25Index] = None,
        concept_graph: Optional[ConceptGraphIndex] = None,
    ):
        self.documents = documents
        self.doc_map = {d.document_id: d for d in documents}
        self.doc_translations = doc_translations
        self.doc_concepts = doc_concepts
        self.encoder = encoder
        self.embedding_cache = embedding_cache
        self.bm25_index = bm25_index
        self.concept_graph = concept_graph
        self._doc_embeddings: Optional[np.ndarray] = None
        self._doc_ids = [d.document_id for d in documents]

    def build_indexes(self, k1: float = 1.5, b: float = 0.75):
        """Index dense representations, BM25 Hindi texts, and inverted concept graph."""
        # 1. Dense Passage Embeddings (using Hindi translated text)
        hindi_texts = [
            self.doc_translations[d.document_id].translated_text
            if d.document_id in self.doc_translations
            else d.lecture
            for d in self.documents
        ]
        self._doc_embeddings = self.embedding_cache.get_or_encode_passages(
            hindi_texts, self.encoder
        )

        # 2. BM25 Index
        self.bm25_index = BM25Index(k1=k1, b=b)
        self.bm25_index.index_documents(self._doc_ids, hindi_texts)

        # 3. Concept Graph Index
        self.concept_graph = ConceptGraphIndex()
        for d in self.documents:
            if d.document_id in self.doc_concepts:
                self.concept_graph.add_document_concepts(d.document_id, self.doc_concepts[d.document_id])

    def retrieve_single(
        self,
        query: QueryRecord,
        query_text: str,
        query_hi_text: Optional[str] = None,
        query_concepts: Optional[QueryConceptRecord] = None,
        system_id: str = "R0",
        top_k: int = 100,
        output_k: int = 10,
        alpha: float = 0.6,
        beta: float = 0.05,
        w_text: float = 0.5,
        w_concept: float = 0.3,
        w_meta: float = 0.2,
    ) -> Tuple[List[str], List[ExplainabilityTrace], Dict[str, Any]]:
        """
        Execute retrieval for a single query using the specified system strategy.
        Returns:
            (ranked_doc_ids, explainability_traces, profiling_metrics)
        """
        start_time = time.perf_counter()

        # Dense Query Embedding & Cosine Similarity
        q_vec = self.embedding_cache.get_or_encode_queries([query_text], self.encoder)[0]
        # Vectors are normalized float32 -> dot product equals cosine similarity
        dense_scores_all = np.dot(self._doc_embeddings, q_vec)
        # Min-max scale dense scores to [0, 1]
        min_d, max_d = float(np.min(dense_scores_all)), float(np.max(dense_scores_all))
        if max_d > min_d:
            norm_dense_all = (dense_scores_all - min_d) / (max_d - min_d)
        else:
            norm_dense_all = np.zeros_like(dense_scores_all)

        doc_dense_map = {doc_id: float(norm_dense_all[idx]) for idx, doc_id in enumerate(self._doc_ids)}

        # BM25 Scores (using Hindi translated query or fallback to query_text)
        bm25_q_text = query_hi_text or query_text
        bm25_results = self.bm25_index.query(bm25_q_text, top_k=len(self._doc_ids))
        doc_bm25_map = {doc_id: score for doc_id, score in bm25_results}

        # Concept Overlap Scores
        concept_candidates = []
        if self.concept_graph and query_concepts:
            concept_candidates = self.concept_graph.get_candidate_documents(query_concepts, candidate_k=len(self._doc_ids))
        doc_concept_map = {doc_id: score for doc_id, score in concept_candidates}

        # Curriculum Metadata Scores
        q_grade_val = parse_grade_to_int(query.grade)
        doc_meta_map = {}
        for d in self.documents:
            m_score = 0.0
            if query.subject and query.subject.lower() == d.subject.lower():
                m_score += 0.4
            if query.topic and query.topic.lower() == d.topic.lower():
                m_score += 0.3
            if query.category and query.category.lower() == d.category.lower():
                m_score += 0.2
            if query.skill and query.skill.lower() == d.skill.lower():
                m_score += 0.1
            doc_meta_map[d.document_id] = m_score

        # Candidate selection and scoring based on system
        num_encoded_candidates = len(self._doc_ids)
        candidate_pool = self._doc_ids

        final_scores: Dict[str, float] = {}

        if system_id == "R0":  # Dense raw
            for d_id in candidate_pool:
                final_scores[d_id] = doc_dense_map.get(d_id, 0.0)

        elif system_id == "R1":  # BM25
            for d_id in candidate_pool:
                final_scores[d_id] = doc_bm25_map.get(d_id, 0.0)

        elif system_id == "R2":  # Hybrid
            for d_id in candidate_pool:
                d_score = doc_dense_map.get(d_id, 0.0)
                b_score = doc_bm25_map.get(d_id, 0.0)
                final_scores[d_id] = alpha * d_score + (1.0 - alpha) * b_score

        elif system_id == "R3":  # Grade-aware dense
            for d_id in candidate_pool:
                d_score = doc_dense_map.get(d_id, 0.0)
                d_grade_val = parse_grade_to_int(self.doc_map[d_id].grade)
                grade_dist = abs(q_grade_val - d_grade_val)
                final_scores[d_id] = d_score - beta * grade_dist

        elif system_id == "R4":  # Metadata-aware
            for d_id in candidate_pool:
                d_score = doc_dense_map.get(d_id, 0.0)
                m_score = doc_meta_map.get(d_id, 0.0)
                final_scores[d_id] = 0.7 * d_score + 0.3 * m_score

        elif system_id == "R5":  # Bilingual concept fusion
            for d_id in candidate_pool:
                d_score = doc_dense_map.get(d_id, 0.0)
                c_score = doc_concept_map.get(d_id, 0.0)
                m_score = doc_meta_map.get(d_id, 0.0)
                final_scores[d_id] = w_text * d_score + w_concept * c_score + w_meta * m_score

        elif system_id == "R6":  # Concept-first candidate generation + dense reranking
            pool_set = set()
            candidate_list = []
            if concept_candidates:
                for doc_id, _ in concept_candidates:
                    if doc_id not in pool_set:
                        pool_set.add(doc_id)
                        candidate_list.append(doc_id)
            # Pad with BM25 / full corpus candidates if fewer than output_k
            for doc_id, _ in bm25_results:
                if doc_id not in pool_set and len(candidate_list) < max(top_k, output_k):
                    pool_set.add(doc_id)
                    candidate_list.append(doc_id)
            for doc_id in self._doc_ids:
                if doc_id not in pool_set and len(candidate_list) < max(top_k, output_k):
                    pool_set.add(doc_id)
                    candidate_list.append(doc_id)

            candidate_pool = candidate_list[:top_k]
            num_encoded_candidates = len(candidate_pool)
            for d_id in candidate_pool:
                final_scores[d_id] = doc_dense_map.get(d_id, 0.0)

        # Sort ranked list
        ranked_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked_items[:output_k]]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Build Explainability Traces for output candidates
        traces = []
        for rank, (doc_id, score) in enumerate(ranked_items[:output_k], start=1):
            doc_obj = self.doc_map[doc_id]
            trans_obj = self.doc_translations.get(doc_id)
            doc_c_obj = self.doc_concepts.get(doc_id)

            trace = create_explainability_trace(
                query=query,
                document=doc_obj,
                rank=rank,
                dense_score=doc_dense_map.get(doc_id, 0.0),
                bm25_score=doc_bm25_map.get(doc_id, 0.0),
                concept_score=doc_concept_map.get(doc_id, 0.0),
                metadata_score=doc_meta_map.get(doc_id, 0.0),
                final_score=score,
                query_concepts=query_concepts,
                doc_concepts=doc_c_obj,
                translation=trans_obj,
                embedding_model=self.encoder.model_name,
            )
            traces.append(trace)

        profiling = {
            "latency_ms": latency_ms,
            "num_candidates": num_encoded_candidates,
            "system_id": system_id,
        }
        return ranked_doc_ids, traces, profiling
