# Table10 Failure Categories `[EXPLORATORY]`

| failure_category                                                  |   occurrence_count |   percentage_of_failures |
|:------------------------------------------------------------------|-------------------:|-------------------------:|
| 1. Raw text succeeds and concepts fail                            |                  0 |                        0 |
| 2. Concepts improve rank                                          |                  0 |                        0 |
| 3. Hindi concepts help while English concepts do not              |                  0 |                        0 |
| 4. English concepts help while Hindi concepts do not              |                  0 |                        0 |
| 5. Translation provider disagreement                              |                  0 |                        0 |
| 6. Encoder disagreement                                           |                  0 |                        0 |
| 7. Correct document missing from concept candidate pool           |                  1 |                      100 |
| 8. BM25 succeeds while dense retrieval fails                      |                  0 |                        0 |
| 9. Dense succeeds while BM25 fails                                |                  0 |                        0 |
| 10. Grade metadata improves suitability but hurts retrieval       |                  0 |                        0 |
| 11. Concept generator produces unsupported concepts               |                  0 |                        0 |
| 12. Hindi translation changes scientific meaning                  |                  0 |                        0 |
| 13. LLM judges disagree                                           |                  0 |                        0 |
| 14. Correct document ranked below semantically similar distractor |                  0 |                        0 |
