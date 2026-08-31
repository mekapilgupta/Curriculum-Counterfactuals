# Results Template: Experimental Output Schema

This document outlines the standardized reporting format for empirical retrieval results across all ablations.

## Main Retrieval Results Schema (Table 3)

| System ID | System Name | Translation Provider | Encoder | MRR@10 [95% CI] | Recall@1 | Recall@10 | nDCG@10 | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| R0 | Dense Raw Text | IndicTrans2 | multilingual-e5-base | -- | -- | -- | -- | -- |
| R1 | BM25 Translated | IndicTrans2 | N/A | -- | -- | -- | -- | -- |
| R2 | Hybrid Dense+BM25 | IndicTrans2 | multilingual-e5-base | -- | -- | -- | -- | -- |
| R3 | Grade-Aware Dense | IndicTrans2 | multilingual-e5-base | -- | -- | -- | -- | -- |
| R4 | Metadata-Aware | IndicTrans2 | multilingual-e5-base | -- | -- | -- | -- | -- |
| R5 | Bilingual Concept Fusion | IndicTrans2 | multilingual-e5-base | -- | -- | -- | -- | -- |
| R6 | Concept-First Candidate Gen | IndicTrans2 | multilingual-e5-base | -- | -- | -- | -- | -- |

## Statistical Significance Schema (Grouped Paired Bootstrap)

- **Baseline System**: R0 (Dense Raw Text)
- **Treatment System**: R5 (Bilingual Concept Fusion)
- **Bootstrap Unit**: Source Lecture Group (`target_document_id`)
- **Number of Replicates**: 2,000
- **Reported Fields**: Absolute Difference, 95% Confidence Interval, Empirical Two-Sided $p$-value, Cohen's $d$.
