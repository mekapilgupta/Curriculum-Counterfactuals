# Experiment Protocol & Execution Steps

## 1. Environment & Setup

Create virtual environment and install dependencies:
```bash
python -m venv .venv
# Activate virtual environment
# On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

Configure `.env` with OpenRouter keys and model selections:
```bash
cp .env.example .env
```

## 2. Pipeline Execution Order

### Step 1: Dataset Inspection & Preprocessing
```bash
python -m curriculum_retrieval.cli inspect-dataset --dataset-name derek-thomas/ScienceQA
python -m curriculum_retrieval.cli prepare-data --config configs/default.yaml
python -m curriculum_retrieval.cli validate-data
python -m curriculum_retrieval.cli make-splits --config configs/default.yaml
```

### Step 2: Translation & Concept Extraction
```bash
# Offline translation via IndicTrans2
python -m curriculum_retrieval.cli translate --provider offline --config configs/default.yaml

# Concept extraction (OpenRouter or heuristic fallback)
python -m curriculum_retrieval.cli generate-concepts --provider heuristic --config configs/default.yaml
```

### Step 3: Multi-System Retrieval Experiments
```bash
# Baseline Dense Raw Text (R0)
python -m curriculum_retrieval.cli retrieve --system R0 --config configs/default.yaml

# BM25 Lexical (R1)
python -m curriculum_retrieval.cli retrieve --system R1 --config configs/default.yaml

# Hybrid Dense + BM25 (R2)
python -m curriculum_retrieval.cli retrieve --system R2 --config configs/default.yaml

# Bilingual Concept Fusion (R5)
python -m curriculum_retrieval.cli retrieve --system R5 --config configs/default.yaml

# Concept-First Candidate Generation + Dense Reranking (R6)
python -m curriculum_retrieval.cli retrieve --system R6 --config configs/default.yaml
```

### Step 4: Human Evaluation & Statistical Reporting
```bash
# Export stratified human audit samples (200 pairs)
python -m curriculum_retrieval.cli export-human-eval --n 200 --seed 42 --output data/annotations/human_eval.jsonl

# Generate all Tables (1-10) and Figures (1-6)
python -m curriculum_retrieval.cli report --config configs/default.yaml
```
