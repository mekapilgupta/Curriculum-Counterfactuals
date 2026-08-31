# Curriculum-Counterfactuals: Reproducible Multilingual Educational Retrieval Research Pipeline

[![CI Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)](#testing)
[![License: CC-BY-NC-SA-4.0](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

This repository provides a reproducible, auditable research pipeline for cross-lingual educational retrieval over translated K-12 science explanations, investigating whether bilingual concept metadata and provenance-aware hybrid retrieval recover retrieval quality beyond raw translated text and standard multilingual embedding baselines.

---

## 1. Research Question

> **In English-to-Hindi retrieval over translated K-12 science explanations, does adding bilingual concept metadata improve answer-supporting retrieval beyond raw translated text, and does the improvement remain stable across translation systems, embedding models, query formulations, and retrieval strategies?**

### Scope Boundaries & Ethical Assertions
- The Hindi corpus comprises **machine-translated educational lecture texts** (derived from ScienceQA English lectures); it is **not** native Hindi textbook content or an authentic Indian curriculum corpus.
- Ground truth query-document relevance labels represent **weak supervision** inherited from ScienceQA question-lecture pairings.
- Grade level is utilized strictly as **structured metadata**, not as an exhaustive cognitive or pedagogical measure.
- The concept graph functions as an **inverted retrieval index**, not an independently adjudicated educational knowledge graph.
- Dual LLM judgments provide secondary exploratory quality metrics and must be paired with human evaluation.

---

## 2. Pipeline Overview

```
 [English ScienceQA Lectures]
               │
               ▼
   [Offline IndicTrans2 / OpenRouter LLMs]
               │
               ▼
   [Translated Hindi Corpus]
               │
               ├────────────────────────────────────────┐
               ▼                                        ▼
   [Bilingual Concepts (EN + HI)]          [Dense & BM25 Indexes]
   - English & Hindi Labels                - multilingual-e5-base
   - Aliases & Evidence Spans              - bge-m3
   - Strict Leakage Isolation              - BM25Plus Multi-Lingual
               │                                        │
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
          [Provenance-Aware Hybrid Retrieval Systems (R0 - R6)]
                                   │
                                   ▼
          [Auditable Evaluation, Bootstrap CIs & Explainability Traces]
```

---

## 3. Retrieval Systems (R0 – R6)

- **R0 (Dense Raw Text)**: English query matched against translated Hindi document embeddings using cosine similarity.
- **R1 (BM25 Translated Query)**: Hindi translated question matched against Hindi documents via BM25Okapi/BM25Plus.
- **R2 (Hybrid Dense+BM25)**: Normalized convex score fusion ($\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$).
- **R3 (Grade-Aware Dense)**: Penalizes documents with grade distance from the query.
- **R4 (Metadata-Aware Retrieval)**: Structured subject, topic, category, and skill field boosting.
- **R5 (Bilingual Concept Fusion)**: Multi-channel score fusion combining text similarity, bilingual concept overlap, and metadata.
- **R6 (Concept-First Candidate Generation)**: Fast candidate filtering ($k=100$) via inverted concept graph followed by dense reranking.

---

## 4. Installation & Environment Setup

```bash
# Clone repository
git clone https://github.com/mekapilgupta/Curriculum-Counterfactuals.git
cd Curriculum-Counterfactuals

# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install in editable mode
pip install -e .
```

### API Configuration
Copy `.env.example` to `.env` and configure your OpenRouter credentials (optional for offline mode):
```bash
cp .env.example .env
```

Key environment variables:
- `OPENROUTER_API_KEY`: OpenRouter API key.
- `OPENROUTER_TRANSLATION_MODEL`: e.g., `google/gemini-2.0-flash-001`.
- `OPENROUTER_CONCEPT_MODEL`: e.g., `google/gemini-2.0-flash-001`.
- `OPENROUTER_JUDGE_MODEL_A`: e.g., `google/gemini-2.0-flash-001`.
- `OPENROUTER_JUDGE_MODEL_B`: e.g., `qwen/qwen-2.5-72b-instruct`.

---

## 5. Quickstart & Verification

### Run End-to-End Synthetic Smoke Test
Runs all data validation, translation, concepts, retrieval (R0-R6), metrics, bootstrap, failure analysis, and table/figure generation in seconds:
```bash
python -m curriculum_retrieval.cli smoke-test
```

### Run Unit & Integration Test Suite
```bash
pytest -v tests/
```

---

## 6. Full Experiment Reproduction

### 1. Data Ingestion & Zero-Leakage Grouped Splits
```bash
python -m curriculum_retrieval.cli inspect-dataset --dataset-name derek-thomas/ScienceQA
python -m curriculum_retrieval.cli prepare-data --config configs/default.yaml
python -m curriculum_retrieval.cli validate-data
python -m curriculum_retrieval.cli make-splits --config configs/default.yaml
```

### 2. Translation & Concept Generation
```bash
# Offline translation with IndicTrans2
python -m curriculum_retrieval.cli translate --provider offline --config configs/default.yaml

# Concept generation
python -m curriculum_retrieval.cli generate-concepts --provider heuristic --config configs/default.yaml
```

### 3. Run Multi-System Retrieval Ablations
```bash
# Dense raw baseline (R0)
python -m curriculum_retrieval.cli retrieve --system R0 --config configs/default.yaml

# Bilingual concept fusion (R5)
python -m curriculum_retrieval.cli retrieve --system R5 --config configs/default.yaml

# Concept-first candidate generation (R6)
python -m curriculum_retrieval.cli retrieve --system R6 --config configs/default.yaml
```

### 4. Human Evaluation Export & Analysis
```bash
# Export stratified 200 sample pairs for annotation
python -m curriculum_retrieval.cli export-human-eval --n 200 --output data/annotations/human_eval.jsonl

# Evaluate completed human annotations
python -m curriculum_retrieval.cli evaluate-human --input data/annotations/human_eval.jsonl
```

### 5. Generate Paper Tables & Figures
```bash
python -m curriculum_retrieval.cli report --config configs/default.yaml
```
Output artifacts are saved in:
- `outputs/tables/`: Tables 1 through 10 in LaTeX (`.tex`), Markdown (`.md`), and CSV (`.csv`).
- `outputs/figures/`: Figures 1 through 6 in 300 DPI publication quality PNG format.
- `outputs/failures/`: Categorized failure cases across 14 failure modes.

---

## 7. Repository Structure

```
Curriculum-Counterfactuals/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   ├── default.yaml
│   └── smoke.yaml
├── data/
│   ├── processed/
│   ├── manifests/
│   ├── translations/
│   ├── concepts/
│   └── annotations/
├── docs/
│   ├── methodology.md
│   ├── dataset.md
│   ├── experiment_protocol.md
│   ├── leakage_policy.md
│   ├── annotation_guidelines.md
│   ├── limitations.md
│   └── results_template.md
├── src/
│   └── curriculum_retrieval/
│       ├── __init__.py
│       ├── schemas.py
│       ├── provenance.py
│       ├── dataset.py
│       ├── splits.py
│       ├── translation.py
│       ├── concepts.py
│       ├── embeddings.py
│       ├── bm25.py
│       ├── graph_index.py
│       ├── retrieval.py
│       ├── explainability.py
│       ├── metrics.py
│       ├── bootstrap.py
│       ├── llm_judging.py
│       ├── human_eval.py
│       ├── failures.py
│       ├── reporting.py
│       ├── smoke.py
│       └── cli.py
├── tests/
│   └── test_pipeline.py
└── outputs/
    ├── smoke/
    ├── tables/
    ├── figures/
    └── failures/
```

---

## 8. Citation & Licensing

ScienceQA source data is distributed under **CC-BY-NC-SA-4.0**. Code in this repository is licensed under the MIT License.
