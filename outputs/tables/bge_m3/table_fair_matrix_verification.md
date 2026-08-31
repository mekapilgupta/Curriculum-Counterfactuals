| Condition_Key                     | Evaluation_Condition                                      |   MRR_at_10 | Rel_Diff_vs_Raw   | Grouped_95_CI    |   P_Value |
|:----------------------------------|:----------------------------------------------------------|------------:|:------------------|:-----------------|----------:|
| 1a_en_raw_reference               | 1. English Raw Text (Monolingual Reference)               |      0.6241 | N/A               | [+0.093, +0.226] |    0.0005 |
| 1b_en_matched_bilingual           | 2. English Text + Bilingual Concepts (Matched Reference)  |      0.6571 | N/A               | [+0.112, +0.272] |    0.0005 |
| 2a_hi_raw_r0                      | 3. Hindi Raw Text Baseline (R0)                           |      0.467  | +0.0%             | [0.000, 0.000]   |    1      |
| 2b_hi_fair_predicted_metadata     | 4. Fair Zero-Shot Predicted Metadata Baseline (R4-Fair)   |      0.4707 | +0.8%             | [-0.018, +0.026] |    0.75   |
| 2c_hi_pure_bilingual_concepts_r5  | 5. Pure Bilingual Concepts Fusion (R5-Pure, Leakage-Free) |      0.5679 | +21.6%            | [+0.037, +0.175] |    0.0005 |
| 2d_hi_concepts_plus_fair_meta     | 6. Bilingual Concepts + Fair Predicted Metadata (R5+Meta) |      0.5504 | +17.9%            | [+0.023, +0.155] |    0.006  |
| 2e_hi_gold_leaked_metadata_oracle | 7. [ORACLE] Gold Leaked Metadata (ScienceQA Row Tags)     |      0.6956 | +49.0%            | [+0.186, +0.273] |    0.0005 |
| 2f_hi_original_leaked_r5          | 8. [LEAKED R5] Original Concepts + Leaked Gold Metadata   |      0.7192 | +54.0%            | [+0.186, +0.324] |    0.0005 |
| 3_bm25_fair_hi_query              | 9. Fair Lexical BM25 (Hindi Query -> Hindi Doc)           |      0.3094 | -33.8%            | [-0.242, -0.074] |    0.001  |