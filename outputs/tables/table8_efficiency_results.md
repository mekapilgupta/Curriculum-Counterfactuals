# Table8 Efficiency Results `[SECONDARY]`

| Retrieval System                  | Candidate Rec@100   | Final Rec@10   |   Latency (ms) |   Candidates Encoded |
|:----------------------------------|:--------------------|:---------------|---------------:|---------------------:|
| Full Dense Rerank (R0)            | 100.0%              | 68.4%          |           48.2 |                 5000 |
| BM25 First + Dense Rerank         | 84.1%               | 71.2%          |            8.4 |                  100 |
| Concept First + Dense Rerank (R6) | 89.6%               | 77.5%          |            6.1 |                  100 |
