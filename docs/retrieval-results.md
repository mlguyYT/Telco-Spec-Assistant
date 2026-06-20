# Retrieval Results

These results compare three retrieval paths over the same 31-question evaluation set for 3GPP TS 38.322, using the same clause-aware chunks and citation metadata.

The evaluation set is intentionally small and focused on one specification, so the numbers show direction rather than statistical precision. The main signal is the retrieval tradeoff: lexical retrieval is strong on exact wording, vector retrieval is strong on paraphrases, and rank-fused hybrid retrieval combines both.

## Setup

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

## Metrics

| Metric | Meaning |
|---|---|
| Answerable recall@5 | Any expected supporting clause appears in the top 5 retrieved chunks for answerable questions. |
| Non-paraphrase recall@5 | Same recall metric on questions without paraphrase tagging. |
| Paraphrase recall@5 | Same recall metric on deliberately reworded questions. |
| Abstention accuracy | Out-of-scope questions return no evidence. |
| Latency p50 / p95 | End-to-end retrieval latency measured by the eval runner. |

## Results

| Retriever | Answerable recall@5 | Non-paraphrase recall@5 | Paraphrase recall@5 | Abstention accuracy | Latency p50 / p95 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.828 | 1.000 | 0.000 | 1.000 | 0.48 ms / 0.61 ms |
| Vertex AI Vector Search | 0.931 | 0.923 | 1.000 | 1.000 | 456 ms / 878 ms |
| Hybrid RRF | 1.000 | 1.000 | 1.000 | 1.000 | 471 ms / 624 ms |

## Interpretation

BM25 is strong when the question wording overlaps with the specification text. It gets full recall on the non-paraphrase subset, but fails the paraphrase subset.

Vertex AI Vector Search recovers the paraphrase subset completely, which is the expected behavior for semantic retrieval. It misses some exact/non-paraphrase cases that BM25 finds.

Hybrid retrieval uses Reciprocal Rank Fusion rather than raw score fusion, because BM25 scores and vector distances are not comparable. The hybrid path keeps the exact-match strength of BM25 and the paraphrase strength of vector retrieval on this evaluation set.

## Operational Notes

The Vector Search endpoint used for this measurement was deleted after the run. Generated vectors and full eval reports are local artifacts under `.data/` and are not committed.

The next retrieval-quality step is to expand the corpus and evaluation set before relying on the third decimal place of these metrics.
