| Condition_Key                     | Evaluation_Condition                                      |   MRR_at_10 | Rel_Diff_vs_Raw   | Grouped_95_CI    |   P_Value |
|:----------------------------------|:----------------------------------------------------------|------------:|:------------------|:-----------------|----------:|
| 1a_en_raw_reference               | 1. English Raw Text (Monolingual Reference)               |      0.6517 | N/A               | [+0.290, +0.411] |    0.0005 |
| 1b_en_matched_bilingual           | 2. English Text + Bilingual Concepts (Matched Reference)  |      0.6746 | N/A               | [+0.309, +0.440] |    0.0005 |
| 2a_hi_raw_r0                      | 3. Hindi Raw Text Baseline (R0)                           |      0.3002 | +0.0%             | [0.000, 0.000]   |    1      |
| 2b_hi_fair_predicted_metadata     | 4. Fair Zero-Shot Predicted Metadata Baseline (R4-Fair)   |      0.3386 | +12.8%            | [+0.002, +0.072] |    0.032  |
| 2c_hi_pure_bilingual_concepts_r5  | 5. Pure Bilingual Concepts Fusion (R5-Pure, Leakage-Free) |      0.4473 | +49.0%            | [+0.106, +0.190] |    0.0005 |
| 2d_hi_concepts_plus_fair_meta     | 6. Bilingual Concepts + Fair Predicted Metadata (R5+Meta) |      0.4626 | +54.1%            | [+0.116, +0.208] |    0.0005 |
| 2e_hi_gold_leaked_metadata_oracle | 7. [ORACLE] Gold Leaked Metadata (ScienceQA Row Tags)     |      0.6177 | +105.8%           | [+0.276, +0.357] |    0.0005 |
| 2f_hi_original_leaked_r5          | 8. [LEAKED R5] Original Concepts + Leaked Gold Metadata   |      0.6875 | +129.0%           | [+0.343, +0.431] |    0.0005 |
| 3_bm25_fair_hi_query              | 9. Fair Lexical BM25 (Hindi Query -> Hindi Doc)           |      0.3094 | +3.0%             | [-0.068, +0.080] |    0.846  |