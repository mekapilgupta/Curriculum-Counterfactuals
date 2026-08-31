"""
Unit and integration test suite for the reproducible multilingual educational retrieval research pipeline.
"""

import json
import numpy as np
import pytest
from curriculum_retrieval.bm25 import BM25Index, tokenize_multilingual
from curriculum_retrieval.bootstrap import compute_paired_grouped_bootstrap
from curriculum_retrieval.concepts import (
    ConceptManager,
    HeuristicConceptGenerator,
    OpenRouterConceptGenerator,
    build_document_variant,
    build_query_variant,
)
from curriculum_retrieval.dataset import validate_data_leakage
from curriculum_retrieval.embeddings import (
    BaseEmbeddingModel,
    EmbeddingCacheManager,
    MockEmbeddingModel,
)
from curriculum_retrieval.explainability import (
    create_explainability_trace,
    extract_matched_concepts,
)
from curriculum_retrieval.graph_index import ConceptGraphIndex
from curriculum_retrieval.human_eval import (
    evaluate_human_annotations,
    export_stratified_human_eval_sample,
)
from curriculum_retrieval.llm_judging import compute_judge_agreement_statistics
from curriculum_retrieval.metrics import (
    compute_all_metrics,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)
from curriculum_retrieval.provenance import (
    compute_dict_hash,
    compute_text_hash,
    set_seed,
)
from curriculum_retrieval.retrieval import RetrievalEngine, parse_grade_to_int
from curriculum_retrieval.schemas import (
    BilingualConcept,
    ConceptRecord,
    ExplainabilityTrace,
    LLMJudgeOutput,
    QueryConceptRecord,
    QueryRecord,
    SourceDocumentRecord,
    TranslationRecord,
)
from curriculum_retrieval.smoke import generate_synthetic_smoke_data
from curriculum_retrieval.splits import create_grouped_splits, verify_split_leakage
from curriculum_retrieval.translation import MockTranslationProvider, TranslationManager


def test_provenance_hashing_and_reproducibility():
    text1 = "Photosynthesis is the process by which green plants make food."
    text2 = "  photosynthesis is the  process by which green plants make food.  "
    h1 = compute_text_hash(text1)
    h2 = compute_text_hash(text2)
    assert h1 == h2, "Text normalization must produce identical SHA-256 hash"

    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert compute_dict_hash(d1) == compute_dict_hash(d2), "Dict hashing must be key-order invariant"


def test_leakage_policy_enforcement():
    docs, queries = generate_synthetic_smoke_data()
    leakage = validate_data_leakage(docs, queries)
    assert leakage["valid"] is True
    assert leakage["error_count"] == 0


def test_split_grouping_zero_leakage():
    docs, queries = generate_synthetic_smoke_data()
    splits, manifest = create_grouped_splits(
        docs, queries, train_fraction=0.6, dev_fraction=0.2, test_fraction=0.2, seed=42
    )
    res = verify_split_leakage(docs, splits)
    assert res["is_leakage_free"] is True
    assert res["leakage_count"] == 0

    # Ensure all target documents are partitioned without overlap
    train_docs = {q.target_document_id for q in splits["train"]}
    test_docs = {q.target_document_id for q in splits["test"]}
    dev_docs = {q.target_document_id for q in splits["dev"]}
    assert train_docs.isdisjoint(test_docs)
    assert train_docs.isdisjoint(dev_docs)
    assert dev_docs.isdisjoint(test_docs)


def test_translation_caching(tmp_path):
    trans_mgr = TranslationManager(cache_dir=tmp_path)
    mock_trans = MockTranslationProvider()
    doc_id = "doc_test_1"
    text = "Electric current flows through conductive wires."

    # First call translates and caches
    rec1 = trans_mgr.get_or_translate(doc_id, text, mock_trans)
    assert rec1.translated_text.startswith("[अनुवाद]")

    # Second call hits cache
    rec2 = trans_mgr.get_or_translate(doc_id, text, mock_trans)
    assert rec1.translated_text_hash == rec2.translated_text_hash


def test_concept_extraction_and_variants(tmp_path):
    concept_mgr = ConceptManager(cache_dir=tmp_path)
    heur = HeuristicConceptGenerator()

    doc = SourceDocumentRecord(
        document_id="doc_bio_1",
        source_dataset="test",
        source_row_id="r1",
        source_split="train",
        source_text_hash="h1",
        lecture="Mitosis is a process of cell division generating identical cells.",
        lecture_length_chars=67,
    )
    rec = concept_mgr.get_or_generate_doc_concepts(doc, heur)
    assert len(rec.concepts) >= 2
    assert rec.generator_provider == "heuristic"

    trans = TranslationRecord(
        document_id="doc_bio_1",
        translation_id="t1",
        source_text_hash="h1",
        target_language="hi",
        translation_provider="mock",
        translation_model="mock",
        translated_text="[अनुवाद] कोशिका विभाजन",
        translated_text_hash="th1",
    )

    v0 = build_document_variant("V0", trans, rec, doc)
    v1 = build_document_variant("V1", trans, rec, doc)
    v3 = build_document_variant("V3", trans, rec, doc)
    v4 = build_document_variant("V4", trans, rec, doc)

    assert v0 == "[अनुवाद] कोशिका विभाजन"
    assert "[Concepts EN]:" in v1
    assert "[Bilingual Concepts]:" in v3
    assert "[Curriculum Metadata]:" in v4


def test_query_variants():
    q = QueryRecord(
        query_id="q1",
        question_text="What happens during cell mitosis?",
        source_row_id="r1",
        source_split="test",
        target_document_id="doc_bio_1",
    )
    heur = HeuristicConceptGenerator()
    qc = QueryConceptRecord(
        query_id="q1",
        source_text_hash="qh1",
        generator_provider="heuristic",
        generator_model="mock",
        concepts=heur.extract_concepts(q.question_text, is_query=True),
    )

    q0 = build_query_variant("Q0", q, None, qc)
    q1 = build_query_variant("Q1", q, None, qc)
    q4 = build_query_variant("Q4", q, None, qc)

    assert q0 == "What happens during cell mitosis?"
    assert "[Concepts]:" in q1
    assert "[Bilingual Concepts]:" in q4


def test_bm25_multilingual_tokenization_and_search():
    tokens = tokenize_multilingual("Gravity and गुरुत्वाकर्षण forces")
    assert "gravity" in tokens
    assert "गुरुत्वाकर्षण" in tokens

    bm25 = BM25Index()
    bm25.index_documents(
        ["doc_1", "doc_2"],
        ["[अनुवाद] Gravity forces pull masses", "[अनुवाद] Photosynthesis in green plants"]
    )
    res = bm25.query("Gravity", top_k=2)
    assert res[0][0] == "doc_1"
    assert res[0][1] > res[1][1]


def test_embedding_normalization():
    mock_enc = MockEmbeddingModel(dimension=32)
    vecs = mock_enc.encode_queries(["Newton's laws of motion"])
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-5


def test_retrieval_systems_execution(tmp_path):
    docs, queries = generate_synthetic_smoke_data()
    trans_mgr = TranslationManager(cache_dir=tmp_path / "trans")
    mock_trans = MockTranslationProvider()
    doc_trans = {d.document_id: trans_mgr.get_or_translate(d.document_id, d.lecture, mock_trans) for d in docs}

    heur = HeuristicConceptGenerator()
    concept_mgr = ConceptManager(cache_dir=tmp_path / "concepts")
    doc_concepts = {d.document_id: concept_mgr.get_or_generate_doc_concepts(d, heur) for d in docs}

    encoder = MockEmbeddingModel()
    emb_cache = EmbeddingCacheManager(cache_dir=tmp_path / "emb")

    engine = RetrievalEngine(
        documents=docs,
        doc_translations=doc_trans,
        doc_concepts=doc_concepts,
        encoder=encoder,
        embedding_cache=emb_cache,
    )
    engine.build_indexes()

    q0 = queries[0]
    qc0 = concept_mgr.get_or_generate_query_concepts(q0, heur)

    for sys_id in ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]:
        ranked, traces, prof = engine.retrieve_single(
            query=q0,
            query_text=q0.question_text,
            query_concepts=qc0,
            system_id=sys_id,
            top_k=len(docs),
            output_k=5,
        )
        assert len(ranked) == 5
        assert len(traces) == 5
        assert isinstance(traces[0], ExplainabilityTrace)
        assert traces[0].rank == 1


def test_evaluation_metrics():
    ranked = ["doc_A", "doc_B", "doc_C", "doc_D"]
    relevant = {"doc_B"}

    mrr = compute_reciprocal_rank(ranked, relevant, k=10)
    assert mrr == 0.5

    rec1 = compute_recall_at_k(ranked, relevant, k=1)
    rec2 = compute_recall_at_k(ranked, relevant, k=2)
    assert rec1 == 0.0
    assert rec2 == 1.0

    ndcg = compute_ndcg_at_k(ranked, relevant, k=10)
    assert ndcg > 0.0

    bundle = compute_all_metrics(ranked, relevant)
    assert bundle["mrr@10"] == 0.5
    assert bundle["recall@5"] == 1.0


def test_paired_grouped_bootstrap():
    baseline = [0.2, 0.4, 0.5, 0.1, 0.3]
    treatment = [0.5, 0.7, 0.8, 0.4, 0.6]
    groups = ["g1", "g1", "g2", "g2", "g3"]

    res = compute_paired_grouped_bootstrap(
        baseline_scores=baseline,
        treatment_scores=treatment,
        group_keys=groups,
        n_replicates=500,
        seed=42,
    )
    assert res["absolute_diff"] == 0.3
    assert res["ci_lower"] > 0.0
    assert res["n_groups"] == 3


def test_judge_agreement_statistics():
    ja = [
        LLMJudgeOutput(answer_support=2, pedagogical_suitability=2, language_quality=2, unsupported_claims=0),
        LLMJudgeOutput(answer_support=1, pedagogical_suitability=1, language_quality=2, unsupported_claims=0),
    ]
    jb = [
        LLMJudgeOutput(answer_support=2, pedagogical_suitability=2, language_quality=2, unsupported_claims=0),
        LLMJudgeOutput(answer_support=1, pedagogical_suitability=2, language_quality=2, unsupported_claims=0),
    ]
    stats = compute_judge_agreement_statistics(ja, jb)
    assert stats["exact_agreement_answer_support"] == 1.0
    assert stats["exact_agreement_pedagogical_suitability"] == 0.5


def test_human_eval_export_and_evaluation(tmp_path):
    records = [
        {
            "query_id": "q1",
            "document_id": "doc1",
            "system_id": "R0",
            "translation_provider": "offline",
            "target_grade": "grade6",
            "question": "What is gravity?",
            "document_text_hi": "[अनुवाद] गुरुत्वाकर्षण",
        }
    ]
    out_jsonl = tmp_path / "human_eval.jsonl"
    out_csv = tmp_path / "human_eval.csv"
    exported = export_stratified_human_eval_sample(records, n_samples=1, output_jsonl=out_jsonl, output_csv=out_csv)
    assert len(exported) == 1
    assert out_jsonl.exists()
    assert out_csv.exists()

    eval_empty = evaluate_human_annotations(out_jsonl)
    assert eval_empty["status"] == "empty_annotations"
