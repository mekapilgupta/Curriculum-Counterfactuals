# Methodology: Provenance-Aware Multilingual Educational Retrieval

## 1. Research Question & Scope

This research investigates:

> **Whether bilingual concept metadata and provenance-aware hybrid retrieval recover cross-lingual educational retrieval quality beyond raw translated text and standard multilingual embedding baselines.**

In English-to-Hindi retrieval over translated K-12 science explanations, does adding bilingual concept metadata improve answer-supporting retrieval beyond raw translated text, and does the improvement remain stable across translation systems, embedding models, query formulations, and retrieval strategies?

## 2. Terminology and Scope Boundaries

- **Corpus Nature**: The Hindi documents in this study represent **translated K-12 science explanations** (derived from the ScienceQA English lecture corpus). They are **not** native Hindi textbook content and **not** an authentic Indian curriculum corpus.
- **Supervision Labels**: Primary query-document relevance labels are **weakly supervised relevance labels** inherited from the original ScienceQA question-to-lecture pairing. They are not human-adjudicated ground truth.
- **Grade Metadata**: Grade annotations are treated strictly as **metadata filters and distance features**, not as an exhaustive psychological or pedagogical difficulty metric.
- **Concept Index**: The concept graph is an **inverted retrieval index**, not a fully validated educational knowledge graph or ontology.

## 3. Architecture & Retrieval Systems

We evaluate 7 retrieval systems across two multilingual encoders (`intfloat/multilingual-e5-base` and `BAAI/bge-m3`):

1. **R0 (Dense Raw Text)**: English query encoded and matched against Hindi translated document vectors via cosine similarity.
2. **R1 (BM25 Translated Query)**: Hindi-translated question matched against Hindi documents using BM25 with multilingual tokenization.
3. **R2 (Hybrid Retrieval)**: Normalized convex score fusion of dense and BM25 scores:
   $$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{dense}} + (1 - \alpha) \cdot \text{Score}_{\text{BM25}}$$
4. **R3 (Grade-Aware Dense Retrieval)**: Penalizes documents distant from query grade level:
   $$\text{Score} = \text{Score}_{\text{dense}} - \beta \cdot |\text{Grade}_{\text{query}} - \text{Grade}_{\text{doc}}|$$
5. **R4 (Metadata-Aware Retrieval)**: Structured field matching boosting documents sharing subject, topic, category, and skill.
6. **R5 (Bilingual Concept Fusion)**: Multi-channel fusion combining text similarity, bilingual concept overlap, and metadata compatibility:
   $$\text{Score} = w_{\text{text}} \cdot \text{Score}_{\text{text}} + w_{\text{concept}} \cdot \text{Score}_{\text{concept}} + w_{\text{meta}} \cdot \text{Score}_{\text{meta}}$$
7. **R6 (Concept-First Candidate Generation + Dense Reranking)**: Uses an inverted concept index to generate a candidate pool ($k=100$), then dense-reranks only the candidate pool to minimize compute latency.

## 4. Document & Query Representation Ablations

- **Document Variants**:
  - **V0**: Hindi translated lecture only
  - **V1**: Hindi lecture + English concepts
  - **V2**: Hindi lecture + Hindi concepts
  - **V3**: Hindi lecture + bilingual English-Hindi concepts
  - **V4**: Hindi lecture + bilingual concepts + ScienceQA curriculum metadata
- **Query Variants**:
  - **Q0**: Original English question
  - **Q1**: English question + English concepts
  - **Q2**: Hindi-translated question
  - **Q3**: English question + Hindi concepts
  - **Q4**: English question + bilingual concepts

## 5. Statistical Rigor

All primary and secondary comparisons report:
- **Primary Metric**: MRR@10 (Mean Reciprocal Rank at cutoff 10).
- **Paired Grouped Bootstrap**: 2,000 bootstrap iterations resampled at the source lecture / cluster group level to produce 95% Confidence Intervals and empirical $p$-values.
- **Effect Sizes**: Paired Cohen's $d$.
