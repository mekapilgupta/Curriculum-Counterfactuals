| Encoder              | Translation   | Hypothesis_Test                              |   Score_Diff | Paired_95_CI     |   P_Value | Significant   |
|:---------------------|:--------------|:---------------------------------------------|-------------:|:-----------------|----------:|:--------------|
| multilingual-e5-base | qwen          | H1: Bilingual Concepts vs. Hindi Raw         |       0.1471 | [+0.106, +0.190] |    0.0005 | True          |
| multilingual-e5-base | qwen          | H2: Bilingual Concepts vs. Full English      |      -0.0995 | [-0.167, -0.033] |    0.003  | True          |
| multilingual-e5-base | qwen          | H3: Full English + Concepts vs. Full English |       0.066  | [+0.027, +0.109] |    0.001  | True          |
| multilingual-e5-base | qwen          | H4: Bilingual Concepts vs. EN-Only Concepts  |      -0.0064 | [-0.029, +0.012] |    0.556  | False         |
| multilingual-e5-base | gemini        | H1: Bilingual Concepts vs. Hindi Raw         |       0.1104 | [+0.065, +0.161] |    0.0005 | True          |
| multilingual-e5-base | gemini        | H2: Bilingual Concepts vs. Full English      |      -0.0137 | [-0.080, +0.057] |    0.674  | False         |
| multilingual-e5-base | gemini        | H3: Full English + Concepts vs. Full English |       0.0451 | [+0.005, +0.092] |    0.023  | True          |
| multilingual-e5-base | gemini        | H4: Bilingual Concepts vs. EN-Only Concepts  |       0.0011 | [-0.012, +0.015] |    0.872  | False         |
| bge-m3               | qwen          | H1: Bilingual Concepts vs. Hindi Raw         |       0.0835 | [+0.030, +0.141] |    0.001  | True          |
| bge-m3               | qwen          | H2: Bilingual Concepts vs. Full English      |       0.0138 | [-0.036, +0.068] |    0.603  | False         |
| bge-m3               | qwen          | H3: Full English + Concepts vs. Full English |       0.0507 | [-0.002, +0.109] |    0.057  | False         |
| bge-m3               | qwen          | H4: Bilingual Concepts vs. EN-Only Concepts  |       0.0022 | [-0.010, +0.014] |    0.7    | False         |
| bge-m3               | gemini        | H1: Bilingual Concepts vs. Hindi Raw         |       0.0835 | [+0.030, +0.141] |    0.001  | True          |
| bge-m3               | gemini        | H2: Bilingual Concepts vs. Full English      |       0.0138 | [-0.036, +0.068] |    0.603  | False         |
| bge-m3               | gemini        | H3: Full English + Concepts vs. Full English |       0.0507 | [-0.002, +0.109] |    0.057  | False         |
| bge-m3               | gemini        | H4: Bilingual Concepts vs. EN-Only Concepts  |       0.0022 | [-0.010, +0.014] |    0.7    | False         |