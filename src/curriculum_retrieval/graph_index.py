"""
Inverted concept index and candidate pool generation for concept-first retrieval (R6).
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
from curriculum_retrieval.schemas import ConceptRecord, QueryConceptRecord


class ConceptGraphIndex:
    """
    Lightweight inverted concept index mapping concepts and aliases to candidate documents.
    Explicitly tracks:
    - concept_id -> doc_ids
    - term (en/hi) -> concept_ids -> doc_ids
    """

    def __init__(self):
        self.concept_to_docs: Dict[str, Set[str]] = defaultdict(set)
        self.term_to_concepts: Dict[str, Set[str]] = defaultdict(set)
        self.doc_concepts: Dict[str, ConceptRecord] = {}

    def add_document_concepts(self, doc_id: str, concept_record: ConceptRecord):
        self.doc_concepts[doc_id] = concept_record
        for concept in concept_record.concepts:
            cid = concept.concept_id
            self.concept_to_docs[cid].add(doc_id)

            # Index labels and aliases
            terms = [concept.label_en.lower(), concept.label_hi.lower()]
            terms.extend([a.lower() for a in concept.aliases_en])
            terms.extend([a.lower() for a in concept.aliases_hi])

            for term in terms:
                if term.strip():
                    self.term_to_concepts[term.strip()].add(cid)

    def get_candidate_documents(
        self,
        query_concepts: QueryConceptRecord,
        candidate_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        Generate candidate document IDs based on concept overlap.
        Returns list of (doc_id, concept_overlap_score).
        """
        doc_scores: Dict[str, float] = defaultdict(float)

        for q_concept in query_concepts.concepts:
            target_cids = {q_concept.concept_id}
            
            # Match query terms to indexed concept IDs
            terms = [q_concept.label_en.lower(), q_concept.label_hi.lower()]
            terms.extend([a.lower() for a in q_concept.aliases_en])
            terms.extend([a.lower() for a in q_concept.aliases_hi])
            for t in terms:
                if t.strip() in self.term_to_concepts:
                    target_cids.update(self.term_to_concepts[t.strip()])

            for cid in target_cids:
                matching_docs = self.concept_to_docs.get(cid, set())
                for doc_id in matching_docs:
                    doc_scores[doc_id] += 1.0

        if not doc_scores:
            # Fallback if no exact concept matches
            return []

        # Sort by score descending
        sorted_candidates = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        max_score = sorted_candidates[0][1] if sorted_candidates else 1.0
        return [(doc_id, score / max_score) for doc_id, score in sorted_candidates[:candidate_k]]
