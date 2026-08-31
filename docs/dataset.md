# Dataset Card: Translated ScienceQA for Multilingual Educational Retrieval

## 1. Dataset Source and Provenance

- **Primary Source**: ScienceQA dataset (`derek-thomas/ScienceQA` on Hugging Face / `lupantech/ScienceQA`).
- **License**: CC-BY-NC-SA-4.0.
- **Languages**: Source English (`en`), Target Hindi (`hi`).

## 2. Ingestion & Filtering Rules

A row is retained only if:
1. `question` is non-empty and length $\ge 10$ characters.
2. `lecture` is non-empty and length $\ge 100$ characters.
3. Language is English.
4. Contains valid or deterministic identifier.

## 3. Deduplication Strategy

Educational lectures frequently repeat across multiple questions in ScienceQA.
- Lectures are normalized (whitespace, lowercasing) and hashed using SHA-256 (`source_text_hash`).
- Deduplicated lectures form the unique corpus documents.
- All mapped questions retain a pointer to their target lecture ID.
- In the primary experiment, approximately 5,000 unique lecture documents and 1,000 evaluation queries are processed.

## 4. Leakage Prevention

The following fields are strictly **excluded** from document indexing, concept generation, and retrieval scoring:
- `solution`
- `answer`
- `choices`
- `hint`
- `gold answer`

## 5. Artifact Manifest Paths

- Processed documents: `data/processed/scienceqa_documents.jsonl`
- Evaluation queries: `data/processed/scienceqa_queries.jsonl`
- Deduplication map: `data/processed/duplicate_map.jsonl`
- Dataset manifest: `data/manifests/dataset_manifest.json`
- Grouped split manifest: `data/manifests/split_manifest.json`
