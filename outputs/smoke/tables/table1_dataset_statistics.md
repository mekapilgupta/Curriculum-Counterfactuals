# Table1 Dataset Statistics `[PRIMARY]`

| Stage                        |   Count | Notes                                    |
|:-----------------------------|--------:|:-----------------------------------------|
| Raw Source Rows              |      20 | Original ScienceQA dataset               |
| Usable Non-Empty Lectures    |      10 | Question >=10 chars, Lecture >=100 chars |
| Deduplicated Document Corpus |      10 | SHA-256 normalized text hash             |
| Evaluation Queries           |      20 | Grouped zero-leakage test set            |
