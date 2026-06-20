# Retrieval Results

These results compare three retrieval paths using the same clause-aware chunks and citation metadata.

The current primary benchmark is the 176-question multi-spec evaluation set over 3GPP TS 38.321, TS 38.322, and TS 38.331. The earlier 31-question RLC-only benchmark is kept below as historical context.

## Multi-Spec Setup

| Field | Value |
|---|---|
| Corpus | 3GPP TS 38.321 MAC, TS 38.322 RLC, TS 38.331 RRC |
| Release / version | Rel-19 / v19.2.0 |
| Chunks | 3,227 clause-aware chunks |
| Questions | 176 |
| Answerable questions | 169 |
| Out-of-scope questions | 7 |
| Answerable questions by spec | 60 MAC, 39 RLC, 70 RRC |
| Top K | 5 |
| Vector embedding model | `text-embedding-005` |
| Vector backend | Vertex AI Vector Search |
| Hybrid method | Reciprocal Rank Fusion over BM25 and Vertex results |

## Multi-Spec Results

| Retriever | Answerable recall@5 | MAC recall@5 | RLC recall@5 | RRC recall@5 | Non-paraphrase recall@5 | Paraphrase recall@5 | Abstention accuracy | Latency p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.817 | 0.900 | 0.872 | 0.714 | 0.861 | 0.273 | 1.000 | 18 ms / 26 ms |
| Vertex AI Vector Search | 0.935 | 0.983 | 0.923 | 0.900 | 0.945 | 0.818 | 1.000 | 429 ms / 561 ms |
| Hybrid RRF | 0.947 | 0.967 | 1.000 | 0.900 | 0.964 | 0.727 | 1.000 | 496 ms / 678 ms |

## Multi-Spec Interpretation

BM25 remains fast and strong on exact clause wording, especially MAC and RLC. Its main weaknesses are RRC retrieval over the larger RRC corpus and deliberately reworded paraphrase questions.

Vertex AI Vector Search materially improves the semantic cases: paraphrase recall rises from 0.273 to 0.818, and RRC recall rises from 0.714 to 0.900.

Hybrid RRF has the best overall answerable recall. It recovers all RLC answerable questions in this run and keeps the RRC improvement from vector retrieval, while preserving much of BM25's exact-match strength. Vertex alone is stronger than hybrid on the paraphrase subset in this run, so the next tuning target is the hybrid weighting/ranking behavior for paraphrase-heavy queries.

## RLC-Only Historical Setup

The evaluation set is intentionally small and focused on one specification, so the numbers show direction rather than statistical precision. The main signal is the retrieval tradeoff: lexical retrieval is strong on exact wording, vector retrieval is strong on paraphrases, and rank-fused hybrid retrieval combines both.

| Field | Value |
|---|---|
| Corpus | 3GPP TS 38.322, NR Radio Link Control protocol specification |
| Release / version | Rel-19 / v19.2.0 |
| Chunks | 82 clause-aware chunks |
| Questions | 31 |
| Answerable questions | 29 |
| Out-of-scope questions | 2 |
| Top K | 5 |
| Vector embedding model | `text-embedding-005` |
| Vector backend | Vertex AI Vector Search |
| Hybrid method | Reciprocal Rank Fusion over BM25 and Vertex results |

## Metric Definitions

| Metric | Meaning |
|---|---|
| Answerable recall@5 | Any expected supporting clause appears in the top 5 retrieved chunks for answerable questions. |
| Non-paraphrase recall@5 | Same recall metric on questions without paraphrase tagging. |
| Paraphrase recall@5 | Same recall metric on deliberately reworded questions. |
| Abstention accuracy | Out-of-scope questions return no evidence. |
| Latency p50 / p95 | End-to-end retrieval latency measured by the eval runner. |

## RLC-Only Historical Results

| Retriever | Answerable recall@5 | Non-paraphrase recall@5 | Paraphrase recall@5 | Abstention accuracy | Latency p50 / p95 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.828 | 1.000 | 0.000 | 1.000 | 0.48 ms / 0.61 ms |
| Vertex AI Vector Search | 0.931 | 0.923 | 1.000 | 1.000 | 456 ms / 878 ms |
| Hybrid RRF | 1.000 | 1.000 | 1.000 | 1.000 | 471 ms / 624 ms |

## RLC-Only Historical Interpretation

BM25 is strong when the question wording overlaps with the specification text. It gets full recall on the non-paraphrase subset, but fails the paraphrase subset.

Vertex AI Vector Search recovers the paraphrase subset completely, which is the expected behavior for semantic retrieval. It misses some exact/non-paraphrase cases that BM25 finds.

Hybrid retrieval uses Reciprocal Rank Fusion rather than raw score fusion, because BM25 scores and vector distances are not comparable. The hybrid path keeps the exact-match strength of BM25 and the paraphrase strength of vector retrieval on this evaluation set.

## Operational Notes

The Vector Search endpoints used for these measurements were deleted after the runs. Generated vectors and full eval reports are local artifacts under `.data/` and are not committed.

The next retrieval-quality step is to tune hybrid retrieval against the multi-spec benchmark, especially for paraphrase-heavy queries, before adding generated answers.
