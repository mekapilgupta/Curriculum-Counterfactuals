# Human Evaluation & Annotation Guidelines

## 1. Objective

Annotators evaluate 200 stratified retrieved Hindi explanation passages for English science queries to determine answer-supporting validity, pedagogical suitability, translation accuracy, and concept correctness.

## 2. Evaluation Rubrics

### A. Answer Support (`human_answer_support`):
- **0 = Unhelpful**: Passage does not provide knowledge necessary to answer the question.
- **1 = Partially Useful**: Passage provides relevant background or contextual concepts, but lacks key steps.
- **2 = Directly Supporting**: Passage directly contains the core principles, facts, or definitions required to solve the question.

### B. Pedagogical Suitability (`human_pedagogical_suitability`):
- **0 = Unsuitable**: Explanation complexity or language level is severely mismatched with target grade level.
- **1 = Partially Usable**: Requires substantial pedagogical adaptation for the declared grade level.
- **2 = Highly Suitable**: Appropriate vocabulary, conceptual depth, and clarity for the target grade.

### C. Translation Quality (`human_translation_quality`):
- **0 = Corrupted**: Translation changes scientific meaning or is ungrammatical.
- **1 = Understandable**: Minor stylistic or grammatical issues; scientific concepts remain intact.
- **2 = High Quality**: Natural, grammatically sound Hindi preserving scientific entities and equations.

### D. Concept Correctness (`human_concept_correctness`):
- **0 = Incorrect / Hallucinated**: Concept labels or aliases in Hindi/English do not accurately reflect the science lecture.
- **1 = Partially Correct**: Minor terminological discrepancy.
- **2 = Accurate**: Precise, valid scientific concept naming.
