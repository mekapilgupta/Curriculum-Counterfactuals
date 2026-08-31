Use the following as the final prompt for your coding agent.

```text
You are implementing a reproducible research pipeline for a paper on multilingual educational retrieval.

Do not begin by building a large application. Build a complete, auditable research pipeline that produces experiment outputs, tables, plots, failure cases, and paper-ready methodology notes.

Do not fabricate data, labels, results, model scores, licenses, or citations.

# 1. Final Research Direction

The paper will study:

> Whether bilingual concept metadata and provenance-aware hybrid retrieval recover cross-lingual educational retrieval quality beyond raw translated text and standard multilingual embedding baselines.

Final research question:

> In English-to-Hindi retrieval over translated K-12 science explanations, does adding bilingual concept metadata improve answer-supporting retrieval beyond raw translated text, and does the improvement remain stable across translation systems, embedding models, query formulations, and retrieval strategies?

Secondary questions:

1. Does bilingual concept metadata help more than English-only or Hindi-only metadata?
2. Does it help both dense and hybrid sparse+dense retrieval?
3. Does concept-based candidate generation improve retrieval speed without reducing candidate recall?
4. Are improvements stable across translation provenance buckets?
5. Do retrieval gains persist across `multilingual-e5-base` and `bge-m3`?
6. Can every retrieved result be explained using matched concepts, text spans, provenance, and score components?

This is a controlled retrieval study. It is not a claim that we have built a general educational AI system.

# 2. Important Scope Decision

Use only the ScienceQA dataset for the primary experiment.

Preferred source:

- Hugging Face: `derek-thomas/ScienceQA`
- Original source/repository: `lupantech/ScienceQA`

The code must inspect the dataset schema before processing. Do not assume the exact column names. Support common ScienceQA names such as:

```text
question
lecture
solution
answer
choices
subject
topic
category
skill
grade
image
id
```

Use only:

```text
question
lecture
subject
topic
category
skill
grade
id
split
```

Do not use the following fields for indexing, concept generation, or retrieval scoring:

```text
solution
answer
choices
hint
gold answer
```

Those fields may contain answer leakage.

Use the English question as the retrieval query.

Use the English lecture as the source educational document.

Translate the lecture into Hindi using:

1. An offline IndicTrans2 model.
2. A user-provided Gemini model accessed through OpenRouter.

The translated Hindi documents are the retrieval corpus.

Important wording:

- Call the corpus “translated K-12 science explanations” or “translated educational lecture texts.”
- Do not call it native Hindi textbook content.
- Do not call it an Indian curriculum corpus.
- Do not claim translation quality without auditing it.
- Do not claim that grade metadata fully represents pedagogical difficulty.

# 3. Dataset Size

Use the following default sizes:

## Smoke test

```text
200 source rows
50 query rows
20 concept-generation records
10 human-audit records
```

## Main experiment

```text
5,000 unique lecture documents
1,000 evaluation queries
all available grades after filtering
all available science subjects/topics
```

If fewer than 5,000 unique usable lectures exist, use all available rows and report the actual count.

Do not process the entire 20,000+ dataset initially.

The pipeline must support later expansion through configuration:

```yaml
data:
  max_documents: 5000
  max_queries: 1000
```

# 4. Dataset Filtering

Keep rows only when:

- `question` is non-empty
- `lecture` is non-empty
- lecture length is at least 100 characters
- question length is at least 10 characters
- the row has a valid ID or receives a deterministic generated ID
- the row is text-only or has a usable lecture field
- the language is English or can be confirmed as English

Deduplicate lectures using normalized text hashes.

For every deduplicated lecture, preserve all associated question IDs.

If multiple questions point to the same lecture, the lecture remains one corpus document and all associated questions are considered relevant to it.

Save:

```text
data/processed/scienceqa_documents.jsonl
data/processed/scienceqa_queries.jsonl
data/processed/duplicate_map.jsonl
data/manifests/dataset_manifest.json
```

# 5. Provenance Requirements

Every source row and derived artifact must preserve provenance.

For each document record, store:

```json
{
  "document_id": "...",
  "source_dataset": "derek-thomas/ScienceQA",
  "source_dataset_revision": "...",
  "source_row_id": "...",
  "source_split": "train|validation|test",
  "source_text_hash": "...",
  "question_ids": [],
  "subject": "...",
  "topic": "...",
  "category": "...",
  "skill": "...",
  "grade": "...",
  "source_license": "...",
  "retrieval_date": "...",
  "lecture_length_chars": 0
}
```

For every translation, store:

```json
{
  "document_id": "...",
  "translation_id": "...",
  "source_text_hash": "...",
  "target_language": "hi",
  "translation_provider": "indictrans2|openrouter",
  "translation_model": "...",
  "translation_model_revision": "...",
  "prompt_version": "...",
  "translated_text": "...",
  "translated_text_hash": "...",
  "created_at": "...",
  "translation_status": "success|failed|empty"
}
```

For every concept record, store:

```json
{
  "document_id": "...",
  "source_text_hash": "...",
  "concept_schema_version": "v1",
  "generator_provider": "openrouter|heuristic|manual",
  "generator_model": "...",
  "prompt_version": "...",
  "concepts": [
    {
      "concept_id": "...",
      "label_en": "...",
      "label_hi": "...",
      "aliases_en": [],
      "aliases_hi": [],
      "evidence_span_en": "...",
      "evidence_span_hi": "...",
      "confidence": null
    }
  ],
  "created_at": "..."
}
```

Do not call model confidence “calibrated” unless calibration has been performed.

# 6. Translation

## Offline translation

Implement IndicTrans2 through a configurable Hugging Face model.

Default configuration:

```yaml
translation:
  offline:
    enabled: true
    model_name: ai4bharat/indictrans2-en-indic-1B
    source_language: eng_Latn
    target_language: hin_Deva
    device: auto
    batch_size: 8
    max_length: 512
```

The implementation must support the currently installed IndicTrans2 API or fail with a clear installation message.

The offline translation must be cached by source text hash.

## OpenRouter translation

Implement an OpenRouter translation provider using `httpx` or `requests`.

Do not store API keys in code or configuration files.

Read:

```text
OPENROUTER_API_KEY
OPENROUTER_TRANSLATION_MODEL
```

from environment variables.

The prompt must require:

- Hindi Devanagari output
- no explanation
- no answer addition
- no summarization
- no omission of scientific terms
- preservation of equations and named entities
- strict JSON output

Support retries, rate limiting, timeouts, caching, and failed-request logging.

If the OpenRouter key or model ID is missing, pause before external execution and ask the user for:

```text
OPENROUTER_API_KEY
OPENROUTER_TRANSLATION_MODEL
```

The pipeline must still support offline-only execution.

# 7. AI Concept Generation

Generate concepts from the English lecture only.

The concept-generation input must not contain:

```text
question
answer
solution
choices
target query
retrieval labels
test labels
```

The concept prompt should request:

```json
{
  "concepts": [
    {
      "label_en": "...",
      "label_hi": "...",
      "aliases_en": [],
      "aliases_hi": [],
      "evidence_span_en": "...",
      "evidence_span_hi": "..."
    }
  ]
}
```

Prompt requirements:

- extract only concepts explicitly supported by the lecture
- produce 3 to 8 concepts
- include scientific entities, processes, relations, quantities, and principles
- preserve important terminology
- provide Hindi labels
- do not invent unsupported background concepts
- abstain when uncertain
- return valid JSON only

Use an OpenRouter model configured through:

```text
OPENROUTER_API_KEY
OPENROUTER_CONCEPT_MODEL
```

Support batching, caching, retries, and JSON validation.

If API credentials are missing, provide a deterministic fallback using:

```text
topic
category
skill
subject
```

but mark fallback concepts as:

```text
generator_provider: heuristic
```

Do not present heuristic metadata as AI-generated metadata.

Also create query concepts from the English question, but never use lecture text or answers to generate query concepts.

Store query concepts separately from document concepts.

# 8. Concept Variants

Implement these document representations:

```text
V0: Hindi translated lecture only
V1: Hindi lecture + English concepts
V2: Hindi lecture + Hindi concepts
V3: Hindi lecture + bilingual English-Hindi concepts
V4: Hindi lecture + bilingual concepts + ScienceQA metadata
```

The metadata fields are:

```text
subject
topic
category
skill
grade
```

Do not concatenate fields invisibly. Store every field separately and make weights configurable.

# 9. Query Variants

Implement these query forms:

```text
Q0: original English question
Q1: English question + English query concepts
Q2: Hindi-translated question
Q3: English question + Hindi query concepts
Q4: English question + bilingual query concepts
```

The Hindi query translation must be performed independently of the target lecture.

Use both query translation sources where possible:

```text
IndicTrans2
OpenRouter translation model
```

Record the translation provenance.

# 10. Retrieval Systems

Implement all of the following.

## R0: Dense raw-text retrieval

- English query
- Hindi document
- cosine similarity
- multilingual encoder

## R1: BM25 translated-query retrieval

- Hindi-translated query
- Hindi document
- configurable Hindi tokenization
- BM25

## R2: Hybrid retrieval

Use normalized score fusion:

```text
hybrid_score =
    alpha * dense_score
    + (1 - alpha) * bm25_score
```

Tune `alpha` only on the development set.

## R3: Grade-aware dense retrieval

Use the target grade as input metadata.

Implement:

```text
score =
    dense_score
    - beta * grade_distance
```

If ScienceQA grades are bands rather than integers, normalize them to an ordered grade-band index.

Tune `beta` only on the development set.

## R4: Metadata-aware retrieval

Use subject, topic, category, skill, and grade as structured fields.

Do not let metadata replace answer-support retrieval. Use metadata as an additional score component or filter.

## R5: Bilingual concept fusion

Use separate indexes or score channels:

```text
final_score =
    w_text * text_score
    + w_concept * concept_score
    + w_metadata * metadata_score
```

Tune weights only on development data.

Required comparisons:

```text
raw Hindi text
Hindi text + English concepts
Hindi text + Hindi concepts
Hindi text + bilingual concepts
Hindi text + bilingual concepts + curriculum metadata
```

## R6: Concept-first candidate generation

Build an inverted concept index:

```text
concept_id -> document_ids
Hindi concept/alias -> document_ids
English concept/alias -> document_ids
```

Use the concept index to generate a candidate pool, then dense-rerank only the candidate pool.

Compare:

```text
full-corpus dense retrieval
concept candidate generation + dense reranking
BM25 candidate generation + dense reranking
```

Report:

- candidate recall@100
- final recall@10
- latency
- number of encoded documents
- memory usage where practical

The concept graph must be represented explicitly as:

```text
question -> concept
concept -> document
document -> provenance
document -> curriculum metadata
```

This is a lightweight retrieval graph, not a validated pedagogical knowledge graph.

Do not call edges “prerequisite” unless independently human-adjudicated.

# 11. Embedding Models

Use exactly two embedding models:

```text
intfloat/multilingual-e5-base
BAAI/bge-m3
```

For e5:

- use the documented query prefix
- use the documented passage prefix
- record pooling and normalization

For bge-m3:

- use the documented sentence-embedding procedure
- record pooling and normalization

Make configurable:

```yaml
encoders:
  - name: intfloat/multilingual-e5-base
    query_prefix: "query: "
    passage_prefix: "passage: "
    normalize: true
    batch_size: 32
    max_length: 512

  - name: BAAI/bge-m3
    query_prefix: ""
    passage_prefix: ""
    normalize: true
    batch_size: 16
    max_length: 512
```

Do not compare raw similarity scores across encoders.

All comparisons must be within encoder.

Cache embeddings using:

```text
model name
model revision
text hash
prefix configuration
pooling configuration
normalization configuration
```

# 12. Evaluation Labels

Primary retrieval relevance is derived from the original ScienceQA query-to-lecture association.

For every query:

```text
positive document = deduplicated lecture attached to that question
```

If multiple source questions map to the same lecture, all are relevant to that lecture.

Do not use `solution`, `answer`, or `choices` to produce retrieval labels.

The paper must describe these as:

> weakly supervised query-document relevance labels inherited from the ScienceQA question-lecture association.

Do not call them teacher annotations.

# 13. LLM Suitability Evaluation

Use two independent LLM judges through OpenRouter.

Read:

```text
OPENROUTER_API_KEY
OPENROUTER_JUDGE_MODEL_A
OPENROUTER_JUDGE_MODEL_B
```

Ask both models to judge only the retrieved question-document pair.

Do not provide the gold answer, solution, or system identity.

Each judge must return:

```json
{
  "answer_support": 0,
  "pedagogical_suitability": 0,
  "language_quality": 0,
  "unsupported_claims": 0,
  "reason": "short structured reason"
}
```

Use fixed rubrics:

```text
answer_support:
0 = does not help answer the question
1 = partially useful
2 = directly supports the answer

pedagogical_suitability:
0 = clearly unsuitable or unrelated
1 = usable with substantial adaptation
2 = suitable for the declared grade/level

language_quality:
0 = unusable or seriously corrupted
1 = understandable with issues
2 = clear and grammatical

unsupported_claims:
0 = no apparent unsupported claims
1 = possible unsupported or mistranslated claim
2 = clear unsupported claim
```

The judge must receive the target grade if the dataset has one.

Do not use LLM judgments as the primary retrieval relevance labels.

Store all raw judge outputs.

Calculate:

- judge agreement
- exact agreement
- weighted agreement
- disagreement rate
- average suitability
- average answer-support quality
- translation error rate

Use the LLM evaluation as a secondary educational-quality analysis.

# 14. Human Evaluation File

Create a separate file:

```text
data/annotations/human_eval.jsonl
```

Do not ask humans to label every result.

Create a stratified sample of 200 query-document pairs across:

```text
translation provider
embedding model
retrieval system
grade
subject/topic
concept variant
LLM judge agreement/disagreement
```

Do not sample simply by every tenth row.

Human evaluation fields:

```json
{
  "sample_id": "...",
  "query_id": "...",
  "document_id": "...",
  "translation_provider": "...",
  "system_id": "...",
  "rank": 1,
  "question": "...",
  "document_text_hi": "...",
  "target_grade": "...",
  "llm_judge_a": {},
  "llm_judge_b": {},
  "human_answer_support": null,
  "human_pedagogical_suitability": null,
  "human_translation_quality": null,
  "human_concept_correctness": null,
  "human_pass": null,
  "human_notes": "",
  "annotator_id": ""
}
```

Provide a CSV and JSONL version so the user can inspect and fill it manually.

Create a command:

```bash
python -m curriculum_retrieval.cli export-human-eval \
  --n 200 \
  --seed 42 \
  --output data/annotations/human_eval.jsonl
```

After the human file is filled, provide:

```bash
python -m curriculum_retrieval.cli evaluate-human \
  --input data/annotations/human_eval.jsonl
```

Report:

- human pass rate
- LLM-human agreement
- translation quality
- concept correctness
- disagreements
- failure categories

If the human file is empty, do not claim human validation.

# 15. Data Splits

Use the official ScienceQA split where available.

If the repository has no usable split field, create:

```text
70% train
15% development
15% test
```

Split by normalized lecture hash and source grouping where possible.

The test set must never be used to tune:

```text
alpha
beta
retrieval weights
concept weights
BM25 parameters
grade penalties
thresholds
```

Write:

```text
data/manifests/split_manifest.json
```

Include:

```json
{
  "strategy": "official_or_grouped",
  "seed": 42,
  "train_ids": [],
  "dev_ids": [],
  "test_ids": [],
  "grouping_field": "lecture_hash",
  "created_at": "...",
  "dataset_revision": "..."
}
```

Add leakage tests that verify:

- no duplicated lecture occurs across train/dev/test query groups
- no test labels influence tuning
- no answer/solution text enters document representations
- concept generation does not see question answers
- query concepts do not see document text
- translation provenance is preserved

# 16. Primary Metrics

Use exactly two primary outcomes.

## Primary retrieval outcome

```text
MRR@10
```

Use the source-linked lecture as the positive document.

Primary comparison:

```text
bilingual concept fusion vs raw Hindi translated text
```

The comparison must be made within the same:

```text
translation provider
embedding model
query set
test split
```

## Primary robustness outcome

```text
mean MRR@10 across the two encoders and two translation provenance buckets
```

Report the paired difference between:

```text
bilingual concept fusion
raw translated Hindi text
```

Secondary metrics:

```text
Recall@1
Recall@5
Recall@10
nDCG@10
MRR
precision@k
candidate recall@100
answer-support score
pedagogical suitability score
translation quality score
concept correctness
latency
memory
number of encoded candidates
```

Do not promote secondary metrics to primary status after seeing results.

# 17. Confidence Intervals

Use paired grouped bootstrap.

The bootstrap unit should be the source lecture group or topic group, not individual duplicated judgments.

For each primary comparison report:

```text
absolute difference
95% confidence interval
number of groups
bootstrap seed
effect size
```

Use at least 2,000 bootstrap replicates in the main experiment and 200 in smoke tests.

Do not rely only on p-values.

# 18. Required Ablation Matrix

Run the following minimum matrix:

```text
A. Translation provenance
   1. IndicTrans2 Hindi
   2. OpenRouter/Gemini Hindi

B. Embedding model
   1. multilingual-e5-base
   2. bge-m3

C. Document representation
   1. Hindi text only
   2. Hindi text + English concepts
   3. Hindi text + Hindi concepts
   4. Hindi text + bilingual concepts
   5. Hindi text + bilingual concepts + metadata

D. Query representation
   1. English question
   2. English question + English concepts
   3. Hindi-translated question
   4. English question + Hindi concepts
   5. English question + bilingual concepts

E. Retrieval strategy
   1. Dense
   2. BM25
   3. Hybrid
   4. Grade-aware reranking
   5. Concept-first candidate generation
   6. Concept-first candidate generation + dense reranking
```

Do not run every possible combination blindly.

The main matrix should be:

```text
raw Hindi text vs bilingual concepts
for each translation provider
for each encoder
using the same English query
```

Run the full query and retrieval ablations only on the development set and a selected test subset if computation is limited.

# 19. Explainability Output

For every retrieved result, save:

```json
{
  "query_id": "...",
  "document_id": "...",
  "rank": 1,
  "dense_score": 0.0,
  "bm25_score": 0.0,
  "concept_score": 0.0,
  "metadata_score": 0.0,
  "final_score": 0.0,
  "matched_concepts": [
    {
      "concept_id": "...",
      "query_label": "...",
      "document_label": "...",
      "evidence_span_en": "...",
      "evidence_span_hi": "..."
    }
  ],
  "translation_provider": "...",
  "embedding_model": "...",
  "document_grade": "...",
  "query_grade": "...",
  "subject": "...",
  "topic": "...",
  "source_text_hash": "...",
  "translation_hash": "..."
}
```

The explanation must be based on actual score components and matched fields.

Do not generate post-hoc natural-language explanations and present them as causal explanations.

# 20. Failure Analysis

Automatically produce examples for:

1. Raw text succeeds and concepts fail.
2. Concepts improve rank.
3. Hindi concepts help while English concepts do not.
4. English concepts help while Hindi concepts do not.
5. Translation provider disagreement.
6. Encoder disagreement.
7. Correct document missing from concept candidate pool.
8. BM25 succeeds while dense retrieval fails.
9. Dense succeeds while BM25 fails.
10. Grade metadata improves suitability but hurts retrieval.
11. Concept generator produces unsupported concepts.
12. Hindi translation changes scientific meaning.
13. LLM judges disagree.
14. Correct document is ranked below a semantically similar distractor.

Save:

```text
outputs/failures/failure_cases.jsonl
outputs/failures/failure_summary.csv
```

# 21. Paper-Ready Reports

Generate the following tables:

```text
Table 1: Dataset statistics and filtering
Table 2: Translation and provenance coverage
Table 3: Main retrieval results
Table 4: Bilingual concept ablation
Table 5: Embedding-model robustness
Table 6: Translation-provider comparison
Table 7: Query formulation ablation
Table 8: Hybrid and concept-first efficiency
Table 9: LLM judge and human audit results
Table 10: Failure categories
```

Generate the following figures:

```text
Figure 1: Pipeline overview
Figure 2: MRR@10 by translation provider and encoder
Figure 3: Concept augmentation ablation
Figure 4: Relevance-quality versus pedagogical suitability
Figure 5: Candidate recall versus latency
Figure 6: Provenance-stratified improvement
```

All tables must include:

```text
system
translation provider
encoder
query variant
document variant
split
number of queries
number of documents
metric
confidence interval
```

Mark every table as:

```text
primary
secondary
exploratory
```

# 22. Recommended Repository Structure

Use or adapt:

```text
project/
  README.md
  pyproject.toml
  configs/
    default.yaml
    smoke.yaml
  data/
    raw/
    interim/
    processed/
    translations/
    concepts/
    annotations/
    manifests/
  src/
    curriculum_retrieval/
      __init__.py
      schemas.py
      dataset.py
      provenance.py
      splits.py
      translation.py
      concepts.py
      embeddings.py
      bm25.py
      hybrid.py
      graph_index.py
      retrieval.py
      metrics.py
      bootstrap.py
      llm_judging.py
      human_eval.py
      reporting.py
      cli.py
  scripts/
  tests/
  outputs/
    smoke/
    experiments/
    tables/
    figures/
    failures/
  docs/
    annotation_guidelines.md
    data_card.md
    experiment_protocol.md
    leakage_policy.md
```

# 23. Required Packages

Use the existing project environment where possible.

Expected packages:

```text
datasets
transformers
sentence-transformers
torch
accelerate
numpy
pandas
polars
pyarrow
scikit-learn
rank-bm25
pydantic
pyyaml
orjson
tqdm
httpx
tenacity
joblib
matplotlib
seaborn
scipy
statsmodels
pytest
ruff
```

Use `IndicTransToolkit` or the currently required IndicTrans2 package version if needed.

Pin versions in a lock or requirements file where practical.

# 24. Configuration

Create `configs/default.yaml`:

```yaml
seed: 42

data:
  dataset_name: derek-thomas/ScienceQA
  dataset_revision: null
  max_documents: 5000
  max_queries: 1000
  min_lecture_chars: 100
  languages:
    source: en
    target: hi

split:
  use_official_split: true
  fallback_strategy: grouped
  group_field: lecture_hash
  train_fraction: 0.70
  dev_fraction: 0.15
  test_fraction: 0.15

translation:
  offline:
    enabled: true
    model_name: ai4bharat/indictrans2-en-indic-1B
    batch_size: 8
    max_length: 512
  openrouter:
    enabled: false
    model_name: null
    temperature: 0.0
    max_tokens: 1024

concepts:
  enabled: true
  provider: openrouter
  model_name: null
  min_concepts: 3
  max_concepts: 8
  batch_size: 5

encoders:
  - name: intfloat/multilingual-e5-base
    query_prefix: "query: "
    passage_prefix: "passage: "
    normalize: true
    batch_size: 32
    max_length: 512
  - name: BAAI/bge-m3
    query_prefix: ""
    passage_prefix: ""
    normalize: true
    batch_size: 16
    max_length: 512

retrieval:
  top_k: 100
  output_k: 10
  bm25_k1: 1.5
  bm25_b: 0.75
  hybrid_alpha: null
  grade_penalty_beta: null
  concept_weight: null
  metadata_weight: null
  candidate_generation:
    enabled: true
    candidate_k: 100

judging:
  enabled: false
  model_a: null
  model_b: null
  temperature: 0.0
  candidate_count: 10

evaluation:
  primary_metric: mrr@10
  bootstrap_replicates: 2000
  bootstrap_group: lecture_hash
  confidence_level: 0.95

human_eval:
  sample_size: 200
  seed: 42
```

# 25. Command-Line Interface

Implement:

```bash
python -m curriculum_retrieval.cli inspect-dataset
python -m curriculum_retrieval.cli prepare-data
python -m curriculum_retrieval.cli validate-data
python -m curriculum_retrieval.cli make-splits
python -m curriculum_retrieval.cli translate --provider offline
python -m curriculum_retrieval.cli translate --provider openrouter
python -m curriculum_retrieval.cli generate-concepts
python -m curriculum_retrieval.cli embed
python -m curriculum_retrieval.cli build-index
python -m curriculum_retrieval.cli retrieve
python -m curriculum_retrieval.cli evaluate
python -m curriculum_retrieval.cli bootstrap
python -m curriculum_retrieval.cli export-human-eval
python -m curriculum_retrieval.cli evaluate-human
python -m curriculum_retrieval.cli failure-analysis
python -m curriculum_retrieval.cli report
python -m curriculum_retrieval.cli smoke-test
```

Each command must:

- print the effective configuration
- print input and output paths
- print dataset/model hashes
- record the git commit if available
- use deterministic seeds
- write a JSON run manifest
- refuse to overwrite cached artifacts without `--force`
- fail clearly when required input data is missing

# 26. Tests

Add tests for:

```text
dataset schema validation
duplicate lecture detection
source hash reproducibility
split leakage
answer/solution exclusion
concept prompt exclusion of answers
translation cache behavior
IndicTrans2 provider interface
OpenRouter mock provider
embedding normalization
BM25 ranking
dense ranking
hybrid score fusion
grade penalty
concept score fusion
concept candidate recall
MRR
Recall@k
nDCG
grouped bootstrap
human evaluation file validation
explanation trace completeness
configuration hashing
```

Tests must not use internet access or real API calls.

Use mocked translation, embedding, and OpenRouter providers.

# 27. Smoke Test Dataset

Create a synthetic dataset containing:

```text
10 documents
20 questions
3 grades
multiple subjects/topics
English lectures
mock Hindi translations
English and Hindi concepts
duplicate lecture examples
```

The smoke test must run:

```text
prepare data
validate data
make splits
mock translation
mock concept generation
mock embeddings
BM25
dense retrieval
hybrid retrieval
concept retrieval
metrics
bootstrap
report generation
```

# 28. README Requirements

Document:

- final research question
- dataset source and revision
- what the dataset does and does not represent
- translation methodology
- provenance schema
- concept-generation methodology
- leakage restrictions
- retrieval systems
- embedding models
- metrics
- bootstrap procedure
- human audit procedure
- limitations
- licensing
- API-key setup
- exact commands
- known blockers

Explicitly state:

```text
The Hindi corpus is machine-translated.
The dataset is not native Hindi textbook content.
Source-linked relevance labels are weak supervision.
LLM judgments are secondary and audited.
Human audit is required before claiming translation or concept quality.
Grade is used as metadata, not as a complete measure of pedagogical difficulty.
The concept graph is a retrieval index, not a validated educational knowledge graph.
Two encoders provide robustness evidence, not universal generality.
```

# 29. API Credential Handling

Before making any external OpenRouter call, check for:

```text
OPENROUTER_API_KEY
OPENROUTER_TRANSLATION_MODEL
OPENROUTER_CONCEPT_MODEL
OPENROUTER_JUDGE_MODEL_A
OPENROUTER_JUDGE_MODEL_B
```

If missing, ask the user for the missing values.

Never print the key.

Never store the key in YAML, JSON, logs, output manifests, or git-tracked files.

If the user provides only one model ID, use it only for the requested purpose and mark all other OpenRouter-dependent stages as pending.

# 30. Final Deliverables

After implementation, provide:

1. Changed-file summary.
2. Installation commands.
3. Smoke-test command and result.
4. Dataset inspection result.
5. Exact number of usable ScienceQA rows.
6. Number of unique lectures.
7. Number of translated documents.
8. Number of generated concept records.
9. Number of evaluation queries.
10. Exact model IDs and revisions.
11. Exact translation providers.
12. Test results.
13. Unresolved licensing issues.
14. Missing API credentials.
15. Commands for running the main experiment.
16. Paths to all output tables and figures.
17. A paper-materials directory containing:
    - `methodology.md`
    - `dataset.md`
    - `experiment_protocol.md`
    - `results_template.md`
    - `limitations.md`

Do not write fabricated numerical results into the paper materials.

# 31. Implementation Order

Follow this order:

1. Inspect the existing repository.
2. Inspect the ScienceQA schema and dataset card.
3. Build typed schemas and validation.
4. Build the synthetic smoke test.
5. Implement data filtering and deduplication.
6. Implement split manifests.
7. Implement offline translation.
8. Implement mock and OpenRouter translation providers.
9. Implement concept generation and caching.
10. Implement both embedding models.
11. Implement BM25, dense, hybrid, metadata, and concept retrieval.
12. Implement metrics and grouped bootstrap.
13. Implement explainability traces.
14. Implement LLM judging.
15. Implement human evaluation export.
16. Implement reporting.
17. Run smoke tests.
18. Ask for missing OpenRouter credentials.
19. Run the 5,000-document experiment.
20. Produce paper-ready outputs.

Start by inspecting the repository and reporting what already exists. Then implement the smallest complete vertical slice:

```text
ScienceQA inspection
-> synthetic test
-> validation
-> mock translation
-> mock concepts
-> retrieval
-> metrics
-> report
```

Only expand to real translation, real embeddings, OpenRouter calls, and the full experiment after that vertical slice passes.
```

For the full run, provide the coding agent these values when requested:

```text
OPENROUTER_API_KEY
OPENROUTER_TRANSLATION_MODEL
OPENROUTER_CONCEPT_MODEL
OPENROUTER_JUDGE_MODEL_A
OPENROUTER_JUDGE_MODEL_B
```

Start with the offline pipeline and a 200-row smoke test. Then run the main experiment on approximately **5,000 unique lecture documents and 1,000 evaluation queries**. Do not increase the dataset size until the leakage checks, translation cache, concept validation, and baseline retrieval results are working.
    my github repo https://github.com/mekapilgupta/Curriculum-Counterfactuals.git