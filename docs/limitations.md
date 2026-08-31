# Limitations & Ethical Considerations

1. **Machine-Translated Corpus**:
   - The Hindi educational documents in this study are machine-translated from English ScienceQA lectures using IndicTrans2 and OpenRouter LLMs. They do not represent authentic, native Hindi textbook material written for Indian regional school curricula.

2. **Weak Supervision Relevance Labels**:
   - Retrieval ground truth is derived from the original ScienceQA question-to-lecture metadata associations. These represent weak supervision rather than manual, multi-annotator relevance judgments for all candidate pairs.

3. **Curriculum & Grade Metadata Scope**:
   - Grade level metadata reflects U.S. K-12 grade mappings and serves as structured categorical metadata. It does not encompass all dimensions of student cognitive load or localized pedagogical scaffolding.

4. **Concept Graph as an Index**:
   - The bilingual concept graph constructed in this pipeline functions strictly as an inverted retrieval index. It is not an adjudicated educational knowledge graph or prerequisite ontology.

5. **LLM Judge Auditing**:
   - Secondary LLM judge evaluations are subjective model appraisals and are used strictly as exploratory signals rather than absolute truth without paired human validation.

6. **Encoder Generality**:
   - Evaluations across `intfloat/multilingual-e5-base` and `BAAI/bge-m3` provide controlled robustness evidence across embedding architectures, but do not imply universal performance across all multilingual representations.
