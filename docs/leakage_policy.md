# Leakage Prevention Policy

To maintain scientific integrity and prevent information leakage across splits, representations, and evaluation channels, the following strict policies are enforced across the codebase:

1. **Answer & Solution Exclusion**:
   - `solution`, `answer`, `choices`, `hint`, and `gold answer` fields are forbidden from entering any indexed document representation, query representation, or concept generator prompt.

2. **Concept Extraction Isolation**:
   - Concept extraction on documents sees **only** the English lecture text. It never observes the student question, target answer, or retrieval labels.
   - Concept extraction on queries sees **only** the English question. It never observes the target lecture, solution, or candidate pool.

3. **Split Partitioning Isolation**:
   - Splits are grouped strictly by `lecture_hash` / `document_id`.
   - Zero document overlap is permitted between train, development, and test query partitions.
   - Test split labels are never used to optimize hyperparameters ($\alpha$, $\beta$, BM25 $k_1, b$, fusion weights).

4. **Judge Isolation**:
   - LLM judges receive only the student question, the retrieved Hindi lecture explanation, and target grade metadata.
   - Judges never receive system identities, gold answers, or ground-truth relevance labels.
