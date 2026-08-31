# Table7 Query Ablation `[SECONDARY]`

| Query Form                       | Query Lang   |   MRR@10 |   Recall@10 |
|:---------------------------------|:-------------|---------:|------------:|
| Q0: Original English Question    | EN           |    0.412 |       0.684 |
| Q1: English + English Concepts   | EN           |    0.456 |       0.729 |
| Q2: Hindi Translated Question    | HI           |    0.431 |       0.702 |
| Q3: English + Hindi Concepts     | EN+HI        |    0.478 |       0.751 |
| Q4: English + Bilingual Concepts | EN+HI        |    0.518 |       0.793 |
