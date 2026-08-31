"""
Synthetic smoke test pipeline running an end-to-end audit in seconds.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from rich.console import Console
from curriculum_retrieval.bootstrap import compute_paired_grouped_bootstrap
from curriculum_retrieval.concepts import (
    ConceptManager,
    HeuristicConceptGenerator,
    build_document_variant,
    build_query_variant,
)
from curriculum_retrieval.dataset import validate_data_leakage
from curriculum_retrieval.embeddings import EmbeddingCacheManager, MockEmbeddingModel
from curriculum_retrieval.failures import FailureAnalyzer
from curriculum_retrieval.llm_judging import (
    LLMJudgeEvaluator,
    compute_judge_agreement_statistics,
)
from curriculum_retrieval.metrics import compute_all_metrics
from curriculum_retrieval.provenance import compute_text_hash, save_run_manifest, set_seed
from curriculum_retrieval.reporting import ReportGenerator
from curriculum_retrieval.retrieval import RetrievalEngine
from curriculum_retrieval.schemas import (
    DatasetManifest,
    QueryRecord,
    SourceDocumentRecord,
    SplitManifest,
    TranslationRecord,
)
from curriculum_retrieval.splits import create_grouped_splits, verify_split_leakage
from curriculum_retrieval.translation import MockTranslationProvider, TranslationManager

console = Console()


def generate_synthetic_smoke_data() -> Tuple[List[SourceDocumentRecord], List[QueryRecord]]:
    """Generate deterministic synthetic science dataset for instant smoke testing."""
    synthetic_lectures = [
        ("Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.", "Biology", "Plants", "Life Science", "Photosynthesis", "grade4"),
        ("Gravity is the natural phenomenon by which all things with mass or energy are brought toward one another.", "Physics", "Mechanics", "Physical Science", "Forces and Motion", "grade6"),
        ("An atom consists of a central nucleus surrounded by one or more negatively charged electrons.", "Chemistry", "Atomic Structure", "Physical Science", "Atoms and Molecules", "grade8"),
        ("The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow.", "Earth Science", "Water Cycle", "Earth and Space", "Earth Systems", "grade5"),
        ("Newton's third law of motion states that for every action in nature there is an equal and opposite reaction.", "Physics", "Mechanics", "Physical Science", "Newtonian Laws", "grade7"),
        ("Mitosis is a process where a single cell divides into two identical daughter cells each containing the same number of chromosomes.", "Biology", "Cell Biology", "Life Science", "Cell Division", "grade9"),
        ("Plate tectonics is the scientific theory explaining the movement of the Earth's subterranean plates whose interactions cause earthquakes.", "Earth Science", "Geology", "Earth and Space", "Plate Boundaries", "grade8"),
        ("Chemical reactions involve breaking chemical bonds between reactant molecules and forming new bonds between atoms in product molecules.", "Chemistry", "Reactions", "Physical Science", "Chemical Reactions", "grade10"),
        ("The solar system consists of the Sun and the celestial objects bound to it by gravity, including eight major planets.", "Earth Science", "Astronomy", "Earth and Space", "Solar System", "grade4"),
        ("Electric current is the rate of flow of electric charge past a point or region in an electric circuit.", "Physics", "Electricity", "Physical Science", "Circuits", "grade6"),
    ]

    docs = []
    queries = []

    for i, (lec, subj, top, cat, sk, gr) in enumerate(synthetic_lectures, start=1):
        thash = compute_text_hash(lec)
        doc_id = f"smoke_doc_{i:02d}"
        d = SourceDocumentRecord(
            document_id=doc_id,
            source_dataset="synthetic_smoke",
            source_row_id=f"syn_row_{i}",
            source_split="train",
            source_text_hash=thash,
            question_ids=[f"smoke_q_{i}_a", f"smoke_q_{i}_b"],
            subject=subj,
            topic=top,
            category=cat,
            skill=sk,
            grade=gr,
            lecture_length_chars=len(lec),
            lecture=lec,
        )
        docs.append(d)

        # Question A
        queries.append(
            QueryRecord(
                query_id=f"smoke_q_{i}_a",
                question_text=f"How does {top.lower()} work according to science?",
                source_row_id=f"syn_row_{i}_a",
                source_split="test",
                target_document_id=doc_id,
                grade=gr,
                subject=subj,
                topic=top,
                category=cat,
                skill=sk,
            )
        )
        # Question B
        queries.append(
            QueryRecord(
                query_id=f"smoke_q_{i}_b",
                question_text=f"What are the main scientific principles behind {sk.lower()}?",
                source_row_id=f"syn_row_{i}_b",
                source_split="test",
                target_document_id=doc_id,
                grade=gr,
                subject=subj,
                topic=top,
                category=cat,
                skill=sk,
            )
        )

    return docs, queries


def run_smoke_test(output_base: str | Path = "outputs/smoke") -> Dict[str, Any]:
    """Execute complete end-to-end pipeline on synthetic smoke data."""
    console.print("[bold green]================ STARTING SMOKE TEST ================[/bold green]")
    set_seed(42)
    base_dir = Path(output_base)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare & Validate Data
    docs, queries = generate_synthetic_smoke_data()
    leakage_check = validate_data_leakage(docs, queries)
    assert leakage_check["valid"], f"Leakage detected in synthetic data: {leakage_check['errors']}"
    console.print("[green][OK] Data validation & leakage checks passed.[/green]")

    # 2. Splits & Split Leakage Verification
    split_queries, split_manifest = create_grouped_splits(
        docs, queries, train_fraction=0.6, dev_fraction=0.2, test_fraction=0.2, output_path=base_dir / "split_manifest.json"
    )
    split_check = verify_split_leakage(docs, split_queries)
    assert split_check["is_leakage_free"], f"Split leakage detected: {split_check['leakages']}"
    console.print("[green][OK] Zero-leakage grouped split verification passed.[/green]")

    # 3. Translation
    trans_mgr = TranslationManager(cache_dir=base_dir / "translations")
    mock_trans = MockTranslationProvider()
    doc_translations = {}
    for d in docs:
        tr = trans_mgr.get_or_translate(d.document_id, d.lecture, mock_trans)
        doc_translations[d.document_id] = tr
    console.print(f"[green][OK] Translated {len(doc_translations)} documents via mock provider.[/green]")

    # 4. Concepts
    concept_mgr = ConceptManager(cache_dir=base_dir / "concepts")
    heur_concept = HeuristicConceptGenerator()
    doc_concepts = {}
    for d in docs:
        cr = concept_mgr.get_or_generate_doc_concepts(d, heur_concept)
        doc_concepts[d.document_id] = cr

    query_concepts = {}
    for q in queries:
        qc = concept_mgr.get_or_generate_query_concepts(q, heur_concept)
        query_concepts[q.query_id] = qc
    console.print(f"[green][OK] Extracted concepts for {len(doc_concepts)} documents and {len(query_concepts)} queries.[/green]")

    # 5. Embeddings & Indexes
    encoder = MockEmbeddingModel()
    emb_cache = EmbeddingCacheManager(cache_dir=base_dir / "embeddings")
    retrieval_engine = RetrievalEngine(
        documents=docs,
        doc_translations=doc_translations,
        doc_concepts=doc_concepts,
        encoder=encoder,
        embedding_cache=emb_cache,
    )
    retrieval_engine.build_indexes()
    console.print("[green][OK] Built dense, BM25, and concept graph indexes.[/green]")

    # 6. Retrieval Runs across systems (R0, R1, R2, R5, R6)
    test_queries = split_queries["test"] if split_queries["test"] else queries[:4]
    systems_to_test = ["R0", "R1", "R2", "R5", "R6"]
    system_scores = {s: [] for s in systems_to_test}
    query_group_keys = []
    failure_analyzer = FailureAnalyzer(output_dir=base_dir / "failures")

    all_traces = []
    for q in test_queries:
        qc = query_concepts[q.query_id]
        query_results_map = {}
        for sys_id in systems_to_test:
            ranked_ids, traces, prof = retrieval_engine.retrieve_single(
                query=q,
                query_text=q.question_text,
                query_concepts=qc,
                system_id=sys_id,
                top_k=len(docs),
                output_k=5,
            )
            m = compute_all_metrics(ranked_ids, {q.target_document_id})
            system_scores[sys_id].append(m["mrr@10"])
            query_results_map[sys_id] = ranked_ids
            all_traces.extend(traces)

        query_group_keys.append(q.target_document_id)
        failure_analyzer.analyze_query_runs(
            query_id=q.query_id,
            target_doc_id=q.target_document_id,
            question=q.question_text,
            results_by_system=query_results_map,
            concept_candidates=[d for d, _ in retrieval_engine.concept_graph.get_candidate_documents(qc, candidate_k=10)],
        )

    # 7. Paired Grouped Bootstrap
    boot_res = compute_paired_grouped_bootstrap(
        baseline_scores=system_scores["R0"],
        treatment_scores=system_scores["R5"],
        group_keys=query_group_keys,
        n_replicates=200,
        seed=42,
    )
    console.print(f"[green][OK] Bootstrap CI calculated:[/green] Absolute diff: {boot_res['absolute_diff']}, 95% CI: [{boot_res['ci_lower']}, {boot_res['ci_upper']}]")

    # 8. Dual Judge Evaluation (Mock)
    judge_eval = LLMJudgeEvaluator(model_a="mock-a", model_b="mock-b", cache_dir=base_dir / "annotations")
    judges_a, judges_b = [], []
    for q in test_queries:
        ja, jb = judge_eval.evaluate_pair(
            query_id=q.query_id,
            doc_id=q.target_document_id,
            question=q.question_text,
            doc_text_hi=doc_translations[q.target_document_id].translated_text,
            grade=q.grade,
        )
        judges_a.append(ja)
        judges_b.append(jb)

    judge_stats = compute_judge_agreement_statistics(judges_a, judges_b)
    console.print(f"[green][OK] Judge agreement calculated:[/green] Exact answer agreement: {judge_stats['exact_agreement_answer_support']}")

    # 9. Failures & Reports
    jsonl_fail, csv_fail = failure_analyzer.save_reports()
    fail_df = pd.read_csv(csv_fail)

    rep_gen = ReportGenerator(tables_dir=base_dir / "tables", figures_dir=base_dir / "figures")
    rep_gen.generate_all_tables(
        dataset_manifest={"total_raw_rows": 20, "usable_rows": 10, "unique_documents": 10, "unique_queries": 20},
        experiment_results={},
        bootstrap_results=boot_res,
        judge_stats=judge_stats,
        failure_summary_df=fail_df,
    )
    rep_gen.generate_all_figures()

    # 10. Run Manifest
    save_run_manifest(
        command="smoke-test",
        config={"seed": 42, "synthetic": True},
        inputs={"documents": "synthetic_smoke", "queries": "synthetic_smoke"},
        outputs={"tables": str(base_dir / "tables"), "figures": str(base_dir / "figures")},
        metrics={"mrr_r0": float(np.mean(system_scores["R0"])), "mrr_r5": float(np.mean(system_scores["R5"]))},
        output_path=base_dir / "run_manifest.json",
    )

    console.print("[bold green]================ SMOKE TEST COMPLETED SUCCESSFULLY ================[/bold green]")
    return {
        "status": "success",
        "documents": len(docs),
        "queries": len(queries),
        "tested_queries": len(test_queries),
        "mrr_r0": float(np.mean(system_scores["R0"])),
        "mrr_r5": float(np.mean(system_scores["R5"])),
        "bootstrap": boot_res,
    }
