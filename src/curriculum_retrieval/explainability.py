"""
Explainability trace generation for transparent score attribution and concept matching.
"""

from typing import Dict, List, Optional
from curriculum_retrieval.schemas import (
    ConceptRecord,
    ExplainabilityTrace,
    MatchedConcept,
    QueryConceptRecord,
    QueryRecord,
    SourceDocumentRecord,
    TranslationRecord,
)


def extract_matched_concepts(
    query_concepts: Optional[QueryConceptRecord],
    doc_concepts: Optional[ConceptRecord],
) -> List[MatchedConcept]:
    """Find overlapping concepts between query and retrieved document."""
    if not query_concepts or not doc_concepts:
        return []

    matched = []
    doc_c_map = {c.concept_id: c for c in doc_concepts.concepts}
    doc_labels = {c.label_en.lower(): c for c in doc_concepts.concepts}

    for qc in query_concepts.concepts:
        doc_match = None
        if qc.concept_id in doc_c_map:
            doc_match = doc_c_map[qc.concept_id]
        elif qc.label_en.lower() in doc_labels:
            doc_match = doc_labels[qc.label_en.lower()]

        if doc_match:
            matched.append(
                MatchedConcept(
                    concept_id=doc_match.concept_id,
                    query_label=qc.label_en,
                    document_label=doc_match.label_en,
                    evidence_span_en=doc_match.evidence_span_en,
                    evidence_span_hi=doc_match.evidence_span_hi,
                )
            )
    return matched


def create_explainability_trace(
    query: QueryRecord,
    document: SourceDocumentRecord,
    rank: int,
    dense_score: float = 0.0,
    bm25_score: float = 0.0,
    concept_score: float = 0.0,
    metadata_score: float = 0.0,
    final_score: float = 0.0,
    query_concepts: Optional[QueryConceptRecord] = None,
    doc_concepts: Optional[ConceptRecord] = None,
    translation: Optional[TranslationRecord] = None,
    embedding_model: str = "",
) -> ExplainabilityTrace:
    """Build a rigorous ExplainabilityTrace record from actual retrieval components."""
    matched_concepts = extract_matched_concepts(query_concepts, doc_concepts)

    return ExplainabilityTrace(
        query_id=query.query_id,
        document_id=document.document_id,
        rank=rank,
        dense_score=round(dense_score, 6),
        bm25_score=round(bm25_score, 6),
        concept_score=round(concept_score, 6),
        metadata_score=round(metadata_score, 6),
        final_score=round(final_score, 6),
        matched_concepts=matched_concepts,
        translation_provider=translation.translation_provider if translation else "",
        embedding_model=embedding_model,
        document_grade=document.grade,
        query_grade=query.grade,
        subject=document.subject,
        topic=document.topic,
        source_text_hash=document.source_text_hash,
        translation_hash=translation.translated_text_hash if translation else "",
    )
