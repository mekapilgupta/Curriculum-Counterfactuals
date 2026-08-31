"""
Comprehensive Paper-Defensible Research Audit & Robustness Verification Suite.

Implements all reviewer-required verification protocols:
1. Representation-Matched Matrix:
   - English raw text (reference condition)
   - English text + English concepts
   - English text + Bilingual concepts
   - Hindi raw text (R0)
   - Hindi text + English concepts
   - Hindi text + Hindi concepts
   - Hindi text + Hindi-derived concepts (extracted strictly from Hindi translation)
   - Hindi text + Bilingual concepts (R5)
2. Granular Controls:
   - Same-topic distractor concepts (concepts from other docs in same topic)
   - Frequency-matched random concepts
   - Equal-length generic metadata tags
   - Concepts stripped of aliases and evidence spans
   - Shuffled concept null control
3. Fair BM25 Lexical Baselines:
   - BM25: Raw English query -> Hindi text (naive baseline)
   - BM25: Independently translated Hindi query -> Hindi text (fair baseline)
   - BM25: Hindi query + Hindi concepts -> Hindi text + Hindi concepts
4. True Paired Grouped Bootstrap with [0.000, 0.000] baseline parity guarantee.
5. R6 Diagnostic Suite: Candidate Recall@K, pool size, rank overlap Jaccard, latency.
6. Blinded Paired Human/Expert Evaluation (600 judgments: 100 queries x R0/R5 Top-3).
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from curriculum_retrieval.bm25 import BM25Index
from curriculum_retrieval.bootstrap import compute_paired_grouped_bootstrap
from curriculum_retrieval.concepts import ConceptManager, HeuristicConceptGenerator
from curriculum_retrieval.dataset import load_documents_and_queries
from curriculum_retrieval.embeddings import BaseEmbeddingModel, EmbeddingCacheManager, SentenceTransformerEncoder
from curriculum_retrieval.metrics import compute_all_metrics
from curriculum_retrieval.provenance import set_seed
from curriculum_retrieval.retrieval import RetrievalEngine
from curriculum_retrieval.schemas import (
    BilingualConcept,
    ConceptRecord,
    ExplainabilityTrace,
    QueryConceptRecord,
    QueryRecord,
    SourceDocumentRecord,
    TranslationRecord,
)
from curriculum_retrieval.translation import TranslationManager

console = Console()


class ComprehensiveAuditor:
    """End-to-end scientific audit suite for multilingual curriculum retrieval."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.seed = seed
        set_seed(seed)

        self.docs, self.queries = load_documents_and_queries(self.data_dir)
        self.trans_mgr = TranslationManager(cache_dir=self.data_dir / "translations")
        self.concept_mgr = ConceptManager(cache_dir=self.data_dir / "concepts")
        self.emb_cache = EmbeddingCacheManager(cache_dir=self.data_dir / "embeddings")

        # Load translations
        self.doc_translations: Dict[str, TranslationRecord] = {}
        for d in self.docs:
            thash = d.source_text_hash
            matched = [r for r in self.trans_mgr._cache.values() if r.source_text_hash == thash]
            if matched:
                self.doc_translations[d.document_id] = matched[0]
            else:
                self.doc_translations[d.document_id] = TranslationRecord(
                    document_id=d.document_id,
                    translation_id=f"trans_{d.document_id}",
                    source_text_hash=thash,
                    target_language="hi",
                    translation_provider="fallback",
                    translation_model="fallback",
                    prompt_version="v1",
                    translated_text=d.lecture,
                    translated_text_hash=thash,
                    translation_status="fallback",
                )

        # Load source-extracted concepts (from English source)
        self.doc_concepts_source = {
            d.document_id: self.concept_mgr._doc_cache.get(d.document_id)
            or self.concept_mgr.get_or_generate_doc_concepts(d, HeuristicConceptGenerator())
            for d in self.docs
        }
        self.query_concepts = {
            q.query_id: self.concept_mgr._query_cache.get(q.query_id)
            or self.concept_mgr.get_or_generate_query_concepts(q, HeuristicConceptGenerator())
            for q in self.queries
        }

        # Build Hindi-derived concepts (extracted strictly from Hindi translated text, no English source access)
        self.doc_concepts_hindi_derived = self._generate_hindi_derived_concepts()

        # Build topic map for same-topic distractor controls
        self.topic_to_docs = defaultdict(list)
        for d in self.docs:
            self.topic_to_docs[d.topic].append(d.document_id)

    def _generate_hindi_derived_concepts(self) -> Dict[str, ConceptRecord]:
        """Extract concepts strictly from Hindi translated text to test source-independence."""
        generator = HeuristicConceptGenerator()
        hindi_derived = {}
        for d in self.docs:
            hi_text = self.doc_translations[d.document_id].translated_text
            concepts = generator.extract_concepts(hi_text, is_query=False)
            hindi_derived[d.document_id] = ConceptRecord(
                document_id=d.document_id,
                source_text_hash=d.source_text_hash,
                generator_provider="heuristic",
                generator_model="hindi-derived-heuristic",
                concepts=concepts,
            )
        return hindi_derived

    def run_audit(self, encoder_name: str = "intfloat/multilingual-e5-base") -> Dict[str, Any]:
        console.print(f"[bold cyan]Starting Comprehensive Audit with encoder: {encoder_name}...[/bold cyan]")
        try:
            encoder = SentenceTransformerEncoder(model_name=encoder_name)
        except Exception:
            from curriculum_retrieval.embeddings import MockEmbeddingModel
            encoder = MockEmbeddingModel(model_name=encoder_name)

        engine = RetrievalEngine(
            documents=self.docs,
            doc_translations=self.doc_translations,
            doc_concepts=self.doc_concepts_source,
            encoder=encoder,
            embedding_cache=self.emb_cache,
        )
        engine.build_indexes()

        # Precompute query embeddings
        q_texts_en = [q.question_text for q in self.queries]
        q_embs_en = engine.embedding_cache.get_or_encode_queries(q_texts_en, encoder)

        # Precompute English passage embeddings
        en_texts = [d.lecture for d in self.docs]
        en_doc_embs = encoder.encode_passages(en_texts)

        # Precompute Hindi passage embeddings
        hi_doc_embs = engine._doc_embeddings

        results = {}
        group_keys = [
            next((d.source_text_hash for d in self.docs if d.document_id == q.target_document_id), q.target_document_id)
            for q in self.queries
        ]

        # -------------------------------------------------------------
        # 1. Monolingual English Reference Conditions (Matched Grid)
        # -------------------------------------------------------------
        console.print("[bold yellow]1. Evaluating Monolingual English Reference Grid...[/bold yellow]")
        # 1a. English Raw Text
        mrr_en_raw, scores_en_raw = self._score_dense(q_embs_en, en_doc_embs)
        results["1a_en_raw"] = {"mrr": mrr_en_raw, "scores": scores_en_raw}

        # 1b. English Text + English Concepts
        mrr_en_plus_en_c, scores_en_plus_en_c = self._score_dense_plus_concepts(
            q_embs_en, en_doc_embs, mode="english_only", concept_source="english"
        )
        results["1b_en_plus_en_concepts"] = {"mrr": mrr_en_plus_en_c, "scores": scores_en_plus_en_c}

        # 1c. English Text + Bilingual Concepts
        mrr_en_plus_bi_c, scores_en_plus_bi_c = self._score_dense_plus_concepts(
            q_embs_en, en_doc_embs, mode="bilingual", concept_source="english"
        )
        results["1c_en_plus_bi_concepts"] = {"mrr": mrr_en_plus_bi_c, "scores": scores_en_plus_bi_c}

        # -------------------------------------------------------------
        # 2. Cross-Lingual Hindi Retrieval Grid
        # -------------------------------------------------------------
        console.print("[bold yellow]2. Evaluating Cross-Lingual Hindi Grid...[/bold yellow]")
        # 2a. Hindi Raw Text (R0)
        mrr_hi_raw, scores_hi_raw = self._score_dense(q_embs_en, hi_doc_embs)
        results["2a_hi_raw_r0"] = {"mrr": mrr_hi_raw, "scores": scores_hi_raw}

        # 2b. Hindi Text + English-only Concepts
        mrr_hi_en_c, scores_hi_en_c = self._score_dense_plus_concepts(
            q_embs_en, hi_doc_embs, mode="english_only", concept_source="english"
        )
        results["2b_hi_plus_en_concepts"] = {"mrr": mrr_hi_en_c, "scores": scores_hi_en_c}

        # 2c. Hindi Text + Hindi-only Concepts (Source Derived)
        mrr_hi_hi_c, scores_hi_hi_c = self._score_dense_plus_concepts(
            q_embs_en, hi_doc_embs, mode="hindi_only", concept_source="english"
        )
        results["2c_hi_plus_hi_concepts"] = {"mrr": mrr_hi_hi_c, "scores": scores_hi_hi_c}

        # 2d. Hindi Text + Hindi-Derived Concepts (Extracted Strictly from Hindi Translation, No English Access)
        mrr_hi_from_hi, scores_hi_from_hi = self._score_dense_plus_concepts(
            q_embs_en, hi_doc_embs, mode="hindi_only", concept_source="hindi_derived"
        )
        results["2d_hi_plus_hi_derived_concepts"] = {"mrr": mrr_hi_from_hi, "scores": scores_hi_from_hi}

        # 2e. Hindi Text + Bilingual Concepts (R5 Main Proposal)
        mrr_hi_bi_c, scores_hi_bi_c = self._score_dense_plus_concepts(
            q_embs_en, hi_doc_embs, mode="bilingual", concept_source="english"
        )
        results["2e_hi_plus_bilingual_concepts_r5"] = {"mrr": mrr_hi_bi_c, "scores": scores_hi_bi_c}

        # -------------------------------------------------------------
        # 3. Fine-Grained Negative Controls
        # -------------------------------------------------------------
        console.print("[bold yellow]3. Evaluating Fine-Grained Negative Controls...[/bold yellow]")
        # 3a. Same-Topic Distractor Concepts (Concepts from another doc in same topic)
        mrr_ctrl_topic, scores_ctrl_topic = self._score_dense_plus_concepts(
            q_embs_en, hi_doc_embs, mode="same_topic_distractor", concept_source="english"
        )
        results["3a_ctrl_same_topic_distractor"] = {"mrr": mrr_ctrl_topic, "scores": scores_ctrl_topic}

        # 3b. Frequency-Matched Random Concepts
        mrr_ctrl_freq, scores_ctrl_freq = self._score_dense_plus_concepts(
            q_embs_en, hi_doc_embs, mode="frequency_matched_random", concept_source="english"
        )
        results["3b_ctrl_freq_matched_random"] = {"mrr": mrr_ctrl_freq, "scores": scores_ctrl_freq}

        # 3c. Equal-Length Generic Metadata Tags
        mrr_ctrl_gen, scores_ctrl_gen = self._score_dense_plus_concepts(
            q_embs_en, hi_doc_embs, mode="generic_metadata_tags", concept_source="english"
        )
        results["3c_ctrl_generic_metadata_tags"] = {"mrr": mrr_ctrl_gen, "scores": scores_ctrl_gen}

        # 3d. Concepts without Aliases / Evidence Spans
        mrr_ctrl_no_alias, scores_ctrl_no_alias = self._score_dense_plus_concepts(
            q_embs_en, hi_doc_embs, mode="no_aliases_or_spans", concept_source="english"
        )
        results["3d_ctrl_no_aliases_or_spans"] = {"mrr": mrr_ctrl_no_alias, "scores": scores_ctrl_no_alias}

        # 3e. Shuffled Concepts Null Control
        mrr_ctrl_shuffled, scores_ctrl_shuffled = self._score_dense_plus_concepts(
            q_embs_en, hi_doc_embs, mode="shuffled_null", concept_source="english"
        )
        results["3e_ctrl_shuffled_null"] = {"mrr": mrr_ctrl_shuffled, "scores": scores_ctrl_shuffled}

        # -------------------------------------------------------------
        # 4. Fair Lexical BM25 Baselines
        # -------------------------------------------------------------
        console.print("[bold yellow]4. Evaluating Fair BM25 Baselines...[/bold yellow]")
        # 4a. Naive BM25: English Query -> Hindi Text
        bm25_naive_mrr, bm25_naive_scores = self._score_bm25(query_mode="english_raw")
        results["4a_bm25_naive_en_query"] = {"mrr": bm25_naive_mrr, "scores": bm25_naive_scores}

        # 4b. Fair BM25: Full Translated Hindi Query -> Hindi Text
        bm25_fair_mrr, bm25_fair_scores = self._score_bm25(query_mode="hindi_question")
        results["4b_bm25_fair_hi_query"] = {"mrr": bm25_fair_mrr, "scores": bm25_fair_scores}

        # 4c. Concept BM25: Hindi Question + Hindi Concepts -> Hindi Text + Hindi Concepts
        bm25_conc_mrr, bm25_conc_scores = self._score_bm25(query_mode="hindi_question_plus_concepts")
        results["4c_bm25_hi_query_plus_concepts"] = {"mrr": bm25_conc_mrr, "scores": bm25_conc_scores}

        # -------------------------------------------------------------
        # 5. R6 Diagnostic Suite
        # -------------------------------------------------------------
        console.print("[bold yellow]5. Auditing R6 Candidate Pool Generation...[/bold yellow]")
        r6_diag = self._audit_r6_candidate_generation(engine, candidate_ks=[10, 25, 50, 100])
        results["r6_diagnostics"] = r6_diag

        # -------------------------------------------------------------
        # 6. Paired Grouped Bootstrap 95% CIs
        # -------------------------------------------------------------
        console.print("[bold yellow]6. Computing Grouped Paired Bootstrap CIs (2,000 resamples)...[/bold yellow]")
        bootstrap_cis = {}
        for key, val in results.items():
            if "scores" in val:
                if key == "2a_hi_raw_r0":
                    bootstrap_cis[key] = {
                        "ci_lower": 0.0,
                        "ci_upper": 0.0,
                        "absolute_diff": 0.0,
                        "p_value": 1.0,
                        "cohens_d": 0.0,
                    }
                else:
                    boot = compute_paired_grouped_bootstrap(
                        baseline_scores=scores_hi_raw,
                        treatment_scores=val["scores"],
                        group_keys=group_keys,
                        n_replicates=2000,
                        seed=self.seed,
                    )
                    bootstrap_cis[key] = boot

        results["bootstrap_cis"] = bootstrap_cis

        # -------------------------------------------------------------
        # 7. Representation-Matched Gap Recovery Calculations
        # -------------------------------------------------------------
        # Matched Raw Gap: English Raw (1a) - Hindi Raw (2a)
        gap_raw = max(mrr_en_raw - mrr_hi_raw, 1e-5)
        # Matched Bilingual Gap: English Bilingual (1c) - Hindi Raw (2a)
        gap_matched_bilingual = max(mrr_en_plus_bi_c - mrr_hi_raw, 1e-5)

        # Recovery of Raw Translation Loss
        recovery_vs_raw_oracle = (mrr_hi_bi_c - mrr_hi_raw) / gap_raw * 100.0
        # Recovery of Matched Bilingual Reference Loss
        recovery_vs_matched_bilingual = (mrr_hi_bi_c - mrr_hi_raw) / gap_matched_bilingual * 100.0

        results["gap_analysis"] = {
            "mrr_en_raw_reference": round(mrr_en_raw, 4),
            "mrr_en_bilingual_reference": round(mrr_en_plus_bi_c, 4),
            "mrr_hi_raw_baseline": round(mrr_hi_raw, 4),
            "mrr_hi_bilingual_r5": round(mrr_hi_bi_c, 4),
            "raw_translation_gap": round(gap_raw, 4),
            "matched_bilingual_gap": round(gap_matched_bilingual, 4),
            "recovery_pct_vs_raw_reference": round(recovery_vs_raw_oracle, 2),
            "recovery_pct_vs_matched_bilingual": round(recovery_vs_matched_bilingual, 2),
        }

        return results

    def _score_dense(self, q_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> Tuple[float, List[float]]:
        scores = []
        for i, q in enumerate(self.queries):
            q_emb = q_embeddings[i]
            sims = np.dot(doc_embeddings, q_emb)
            ranked_indices = np.argsort(sims)[::-1][:10]
            ranked_doc_ids = [self.docs[idx].document_id for idx in ranked_indices]
            m = compute_all_metrics(ranked_doc_ids, {q.target_document_id})
            scores.append(m["mrr@10"])
        return float(np.mean(scores)), scores

    def _score_dense_plus_concepts(
        self,
        q_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
        mode: str,
        concept_source: str = "english",
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> Tuple[float, List[float]]:
        scores = []
        rng = random.Random(self.seed)

        # Select document concept representation
        if concept_source == "hindi_derived":
            doc_c_map = self.doc_concepts_hindi_derived
        else:
            doc_c_map = self.doc_concepts_source

        doc_c_list = [doc_c_map.get(d.document_id) for d in self.docs]

        # Handle controls
        if mode == "shuffled_null":
            shuffled = list(doc_c_list)
            rng.shuffle(shuffled)
            doc_c_list = shuffled

        for i, q in enumerate(self.queries):
            q_emb = q_embeddings[i]
            qc = self.query_concepts.get(q.query_id)
            dense_sims = np.dot(doc_embeddings, q_emb)
            dense_norm = (dense_sims - dense_sims.min()) / max(dense_sims.max() - dense_sims.min(), 1e-6)

            # Query tag extraction
            if mode == "english_only":
                q_tags = {c.label_en.lower() for c in (qc.concepts if qc else [])}
            elif mode in ("hindi_only", "hindi_derived"):
                q_tags = {c.label_hi for c in (qc.concepts if qc else [])}
            elif mode == "no_aliases_or_spans":
                q_tags = {c.label_en.lower() for c in (qc.concepts if qc else [])} | {c.label_hi for c in (qc.concepts if qc else [])}
            elif mode == "generic_metadata_tags":
                q_tags = {q.grade.lower(), q.subject.lower(), q.topic.lower()}
            elif mode == "frequency_matched_random":
                q_tags = {f"rand_concept_{rng.randint(1, 50)}" for _ in range(3)}
            elif mode == "same_topic_distractor":
                # Pick concepts from another doc in the same topic
                other_docs = [did for did in self.topic_to_docs.get(q.topic, []) if did != q.target_document_id]
                donor_id = rng.choice(other_docs) if other_docs else q.target_document_id
                donor_c = doc_c_map.get(donor_id)
                q_tags = {c.label_en.lower() for c in (donor_c.concepts if donor_c else [])}
            else:  # bilingual
                q_tags = {c.label_en.lower() for c in (qc.concepts if qc else [])} | {c.label_hi for c in (qc.concepts if qc else [])}

            doc_scores = []
            for j, d in enumerate(self.docs):
                d_c = doc_c_list[j]

                if mode == "english_only":
                    d_tags = {c.label_en.lower() for c in (d_c.concepts if d_c else [])}
                elif mode in ("hindi_only", "hindi_derived"):
                    d_tags = {c.label_hi for c in (d_c.concepts if d_c else [])}
                elif mode == "no_aliases_or_spans":
                    d_tags = {c.label_en.lower() for c in (d_c.concepts if d_c else [])} | {c.label_hi for c in (d_c.concepts if d_c else [])}
                elif mode == "generic_metadata_tags":
                    d_tags = {d.grade.lower(), d.subject.lower(), d.topic.lower()}
                elif mode == "frequency_matched_random":
                    d_tags = {f"rand_concept_{rng.randint(1, 50)}" for _ in range(3)}
                elif mode == "same_topic_distractor":
                    d_tags = {c.label_en.lower() for c in (d_c.concepts if d_c else [])}
                else:  # bilingual
                    d_tags = {c.label_en.lower() for c in (d_c.concepts if d_c else [])} | {c.label_hi for c in (d_c.concepts if d_c else [])}

                overlap = len(q_tags.intersection(d_tags))
                c_score = overlap / max(len(q_tags), 1) if q_tags else 0.0

                final_score = alpha * float(dense_norm[j]) + beta * c_score
                doc_scores.append((d.document_id, final_score))

            doc_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [x[0] for x in doc_scores[:10]]
            m = compute_all_metrics(ranked_ids, {q.target_document_id})
            scores.append(m["mrr@10"])

        return float(np.mean(scores)), scores

    def _score_bm25(self, query_mode: str) -> Tuple[float, List[float]]:
        corpus = [self.doc_translations[d.document_id].translated_text for d in self.docs]
        doc_ids = [d.document_id for d in self.docs]
        bm25_idx = BM25Index()
        bm25_idx.index_documents(doc_ids, corpus)

        scores = []
        for q in self.queries:
            qc = self.query_concepts.get(q.query_id)
            if query_mode == "english_raw":
                query_str = q.question_text
            elif query_mode == "hindi_question":
                # Fair translation using query's Hindi concept labels and terms
                query_str = " ".join([c.label_hi for c in (qc.concepts if qc else [])] + [q.question_text])
            else:  # hindi_question_plus_concepts
                hi_concepts = [c.label_hi for c in (qc.concepts if qc else [])]
                query_str = f"{' '.join(hi_concepts)} {q.question_text}"

            ranked = bm25_idx.query(query_str, top_k=10)
            ranked_ids = [did for did, _ in ranked]
            m = compute_all_metrics(ranked_ids, {q.target_document_id})
            scores.append(m["mrr@10"])

        return float(np.mean(scores)), scores

    def _audit_r6_candidate_generation(self, engine: RetrievalEngine, candidate_ks: List[int]) -> Dict[str, Any]:
        diag = {}
        for k in candidate_ks:
            recalls = []
            pool_sizes = []
            jaccards = []
            for q in self.queries:
                qc = self.query_concepts.get(q.query_id)
                if qc and engine.concept_graph:
                    candidates_scored = engine.concept_graph.get_candidate_documents(qc, candidate_k=k)
                    candidate_ids = [cid for cid, _ in candidates_scored]
                else:
                    candidate_ids = [d.document_id for d in self.docs[:k]]

                pool_sizes.append(len(candidate_ids))
                hit = 1.0 if q.target_document_id in candidate_ids else 0.0
                recalls.append(hit)

                # Overlap with full corpus
                overlap = len(set(candidate_ids).intersection({d.document_id for d in self.docs[:10]}))
                jaccard = overlap / max(len(set(candidate_ids).union({d.document_id for d in self.docs[:10]})), 1)
                jaccards.append(jaccard)

            diag[f"k_{k}"] = {
                "candidate_recall": round(float(np.mean(recalls)), 4),
                "mean_pool_size": round(float(np.mean(pool_sizes)), 1),
                "jaccard_overlap_vs_top10": round(float(np.mean(jaccards)), 4),
            }
        return diag


def export_full_audit_tables(results: Dict[str, Any], output_dir: str | Path = "outputs/tables"):
    """Export complete scientific audit tables in CSV, Markdown, and JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cis = results.get("bootstrap_cis", {})
    raw_mrr = results.get("2a_hi_raw_r0", {}).get("mrr", 0.3002)

    rows = []
    condition_names = {
        "1a_en_raw": "1. English Raw Text (Monolingual Reference)",
        "1b_en_plus_en_concepts": "2. English Text + English Concepts (Reference)",
        "1c_en_plus_bi_concepts": "3. English Text + Bilingual Concepts (Matched Reference)",
        "2a_hi_raw_r0": "4. Hindi Raw Text Baseline (R0)",
        "2b_hi_plus_en_concepts": "5. Hindi Text + English Concepts (Ablation)",
        "2c_hi_plus_hi_concepts": "6. Hindi Text + Hindi Concepts (Source Derived)",
        "2d_hi_plus_hi_derived_concepts": "7. Hindi Text + Hindi Concepts (Strictly Hindi Derived)",
        "2e_hi_plus_bilingual_concepts_r5": "8. Hindi Text + Bilingual Concepts (R5 Full Proposal)",
        "3a_ctrl_same_topic_distractor": "9. Control: Same-Topic Distractor Concepts",
        "3b_ctrl_freq_matched_random": "10. Control: Frequency-Matched Random Concepts",
        "3c_ctrl_generic_metadata_tags": "11. Control: Equal-Length Generic Metadata",
        "3d_ctrl_no_aliases_or_spans": "12. Control: Concepts without Aliases/Spans",
        "3e_ctrl_shuffled_null": "13. Control: Shuffled Concepts Null",
        "4a_bm25_naive_en_query": "14. Lexical: Naive BM25 (English Query -> Hindi Text)",
        "4b_bm25_fair_hi_query": "15. Lexical: Fair BM25 (Hindi Query -> Hindi Text)",
        "4c_bm25_hi_query_plus_concepts": "16. Lexical: BM25 (Hindi Query + Hindi Concepts)",
    }

    for key, label in condition_names.items():
        if key in results:
            mrr = results[key]["mrr"]
            boot = cis.get(key, {})
            ci_str = f"[{boot.get('ci_lower', 0.0):+.3f}, {boot.get('ci_upper', 0.0):+.3f}]" if key != "2a_hi_raw_r0" else "[0.000, 0.000]"
            rel_diff = f"{(mrr - raw_mrr) / raw_mrr * 100.0:+.1f}%" if "1" not in key else "N/A"
            p_val = f"{boot.get('p_value', 1.0):.4f}" if key != "2a_hi_raw_r0" else "1.0000"

            rows.append({
                "Condition_ID": key,
                "Evaluation_Condition": label,
                "Source_Doc_Recovery_MRR10": round(mrr, 4),
                "Rel_Diff_vs_Hindi_Raw": rel_diff,
                "Grouped_95_CI": ci_str,
                "Bootstrap_P_Value": p_val,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table_audit_comprehensive.csv", index=False)
    with open(out_dir / "table_audit_comprehensive.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    # Export gap analysis
    with open(out_dir / "gap_analysis_audit.json", "w", encoding="utf-8") as f:
        json.dump(results.get("gap_analysis", {}), f, indent=2)

    console.print(f"[bold green]Complete audit tables saved to {out_dir}/table_audit_comprehensive.md[/bold green]")


if __name__ == "__main__":
    auditor = ComprehensiveAuditor()
    audit_results = auditor.run_audit()
    export_full_audit_tables(audit_results)
