# Table4 Concept Ablation `[PRIMARY]`

| Variant                        | Concept Source       | MRR@10               |   Recall@10 |   Delta vs V0 |
|:-------------------------------|:---------------------|:---------------------|------------:|--------------:|
| V0: Raw Hindi Text             | None                 | 0.412 [0.389, 0.435] |       0.684 |         0     |
| V1: Hindi + English Concepts   | Extracted EN         | 0.463 [0.438, 0.487] |       0.738 |         0.051 |
| V2: Hindi + Hindi Concepts     | Extracted HI         | 0.482 [0.457, 0.506] |       0.759 |         0.07  |
| V3: Hindi + Bilingual Concepts | Bilingual EN+HI      | 0.521 [0.498, 0.546] |       0.798 |         0.109 |
| V4: V3 + Curriculum Metadata   | Bilingual + Metadata | 0.554 [0.531, 0.578] |       0.832 |         0.142 |
