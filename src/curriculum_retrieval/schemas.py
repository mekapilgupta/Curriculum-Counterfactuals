"""
Pydantic schemas for the reproducible multilingual educational retrieval research pipeline.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class SourceDocumentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str
    source_dataset: str = "derek-thomas/ScienceQA"
    source_dataset_revision: Optional[str] = None
    source_row_id: str
    source_split: str = "train"
    source_text_hash: str
    question_ids: List[str] = Field(default_factory=list)
    subject: str = ""
    topic: str = ""
    category: str = ""
    skill: str = ""
    grade: str = ""
    source_license: str = "CC-BY-NC-SA-4.0"
    retrieval_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    lecture_length_chars: int = 0
    lecture: str


class QueryRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_id: str
    question_text: str
    source_row_id: str
    source_split: str = "test"
    target_document_id: str
    grade: str = ""
    subject: str = ""
    topic: str = ""
    category: str = ""
    skill: str = ""


class TranslationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str
    translation_id: str
    source_text_hash: str
    target_language: str = "hi"
    translation_provider: Literal["indictrans2", "openrouter", "heuristic", "mock"]
    translation_model: str
    translation_model_revision: Optional[str] = None
    prompt_version: str = "v1"
    translated_text: str
    translated_text_hash: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    translation_status: Literal["success", "failed", "empty"] = "success"


class BilingualConcept(BaseModel):
    model_config = ConfigDict(extra="ignore")

    concept_id: str
    label_en: str
    label_hi: str
    aliases_en: List[str] = Field(default_factory=list)
    aliases_hi: List[str] = Field(default_factory=list)
    evidence_span_en: str = ""
    evidence_span_hi: str = ""
    confidence: Optional[float] = None


class ConceptRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str
    source_text_hash: str
    concept_schema_version: str = "v1"
    generator_provider: Literal["openrouter", "heuristic", "manual", "mock"]
    generator_model: str
    prompt_version: str = "v1"
    concepts: List[BilingualConcept] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QueryConceptRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_id: str
    source_text_hash: str
    concept_schema_version: str = "v1"
    generator_provider: Literal["openrouter", "heuristic", "manual", "mock"]
    generator_model: str
    prompt_version: str = "v1"
    concepts: List[BilingualConcept] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MatchedConcept(BaseModel):
    concept_id: str
    query_label: str
    document_label: str
    evidence_span_en: str = ""
    evidence_span_hi: str = ""


class ExplainabilityTrace(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_id: str
    document_id: str
    rank: int
    dense_score: float = 0.0
    bm25_score: float = 0.0
    concept_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    matched_concepts: List[MatchedConcept] = Field(default_factory=list)
    translation_provider: str = ""
    embedding_model: str = ""
    document_grade: str = ""
    query_grade: str = ""
    subject: str = ""
    topic: str = ""
    source_text_hash: str = ""
    translation_hash: str = ""


class LLMJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer_support: int = Field(ge=0, le=2)
    pedagogical_suitability: int = Field(ge=0, le=2)
    language_quality: int = Field(ge=0, le=2)
    unsupported_claims: int = Field(ge=0, le=2)
    reason: str = ""


class HumanEvalRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sample_id: str
    query_id: str
    document_id: str
    translation_provider: str
    system_id: str
    rank: int = 1
    question: str
    document_text_hi: str
    target_grade: str = ""
    llm_judge_a: Dict[str, Any] = Field(default_factory=dict)
    llm_judge_b: Dict[str, Any] = Field(default_factory=dict)
    human_answer_support: Optional[int] = None
    human_pedagogical_suitability: Optional[int] = None
    human_translation_quality: Optional[int] = None
    human_concept_correctness: Optional[int] = None
    human_pass: Optional[bool] = None
    human_notes: str = ""
    annotator_id: str = ""


class DatasetManifest(BaseModel):
    total_raw_rows: int
    usable_rows: int
    unique_documents: int
    unique_queries: int
    min_lecture_chars: int
    min_question_chars: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_dataset: str
    schema_detected: Dict[str, str] = Field(default_factory=dict)


class SplitManifest(BaseModel):
    strategy: str = "official_or_grouped"
    seed: int = 42
    train_ids: List[str] = Field(default_factory=list)
    dev_ids: List[str] = Field(default_factory=list)
    test_ids: List[str] = Field(default_factory=list)
    grouping_field: str = "lecture_hash"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_revision: Optional[str] = None


class RunManifest(BaseModel):
    run_id: str
    command: str
    config_hash: str
    git_commit: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    inputs: Dict[str, str] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
