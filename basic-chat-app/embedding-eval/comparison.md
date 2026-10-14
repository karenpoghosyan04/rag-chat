# Embedding model migration evaluation

## Scope and methodology

The repository contains 41 labeled retrieval questions, not 61: 35 answerable and 6 unanswerable. All 41 were evaluated without changing their labels. The separate Python test suite contains 62 tests after this migration (61 before the added cache-invalidation test). The original source PDF is not present in the repository, so the comparison uses the existing frozen 33-chunk corpus and float32 indexes to preserve the prior methodology.

Cases 5, 10, 15, 20, 25, 30, 35, 37, and 41 form a deterministic held-out set (7 answerable, 2 unanswerable). The other 32 cases (28 answerable, 4 unanswerable) form the calibration set. Candidate thresholds were swept on calibration data only. The objective balanced Recall@5, Precision@5, no-answer accuracy, and overall answerability classification accuracy; the holdout was evaluated once after selection.

`embedding_min_similarity` and `min_score` now have distinct meanings:

- `embedding_min_similarity = 0.630` is a query-level abstention gate over the maximum query-to-chunk cosine similarity. Gemma's unanswerable maxima were 0.583–0.625, while answerable maxima were 0.631–0.799. In embeddings-only retrieval it also filters individual dense candidates.
- `min_score = 0.200` is a per-chunk gate on the normalized cross-encoder score. Labeled relevant candidates ranged from 0.208–0.871. Higher candidates began removing ground-truth evidence; lower candidates did not improve labeled recall.

The relevant-chunk Gemma cosine distribution was 0.608 minimum, 0.667 p25, 0.688 median, and 0.799 maximum. Irrelevant pairs overlapped substantially (0.524–0.761), so pairwise cosine alone is not treated as a relevance classifier. The maximum similarity per query provides the cleaner answerability signal.

## Retrieval quality

Production hybrid results use the same 25-candidate BM25+dense RRF stage, the same `jinaai/jina-reranker-v1-tiny-en` cross-encoder, exact cosine vectors, and the calibrated gates above. Precision@5 uses the existing evaluator's fixed denominator of five.

| Model/configuration | R@1 | R@5 | Precision@5 | MRR@5 | No-answer accuracy | False-positive rate | Answerability accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous BGE report, legacy gates | 57.1% | 91.4% | not recorded | 0.717 | 100.0% | 0.0% | not recorded |
| BGE, independently recalibrated (`0.555` / `0.200`) | 60.0% | 94.3% | 18.9% | 0.745 | 100.0% | 0.0% | 100.0% |
| EmbeddingGemma 2 (`0.630` / `0.200`) | 62.9% | 91.4% | 18.3% | 0.744 | 100.0% | 0.0% | 100.0% |

Gemma's held-out results were 71.4% Recall@1, 100% Recall@5, 20.0% Precision@5, 0.821 MRR@5, 100% no-answer accuracy, and 0% false-positive rate. Because the held-out set has only nine cases and two negatives, these values have high uncertainty and should not be interpreted as production-grade statistical evidence.

For reference, exact dense retrieval before reranking improved from 31.4% to 48.6% Recall@1 and from 91.4% to 97.1% Recall@10, but Recall@5 changed only from 74.3% to 77.1%. The earlier paired bootstrap for exact-dense Recall@5 was +2.86 percentage points with a 95% interval of -14.29 to +20.00 points, so the improvement was not statistically conclusive.

## Performance

| Model/configuration | Document indexing | Avg retrieval latency | p50 / p95 retrieval | Index payload | Peak indexing RSS |
|---|---:|---:|---:|---:|---:|
| BGE baseline | 4.29 s (7.69 chunks/s) | 1,600 ms | 1,561 / 1,709 ms | 49.5 KiB | 886 MiB |
| Gemma, original full-modal one-batch run | 765.10 s (0.04 chunks/s) | not recorded | 3,896 / 4,721 ms | 99.0 KiB | 2,206 MiB |
| Gemma, text-only, batch size 1 | 13.13 s (2.51 chunks/s) | 1,582 ms | 1,573 / 1,796 ms | 99.0 KiB | 1,977 MiB |

The optimized retrieval latency adds the measured 44 ms average single-query Gemma encoding time to the cached-vector retriever measurement. BGE adds its prior 4.6 ms query time. The cross-encoder remains the dominant latency cost. Text-only initialization and avoiding a single padding-heavy document batch reduced Gemma indexing time by about 58x versus the initial Gemma run, but Gemma still indexes about 3.1x slower than BGE, uses about 2.2x the peak memory, and stores twice as many float32 vector bytes (768 versus 384 dimensions).

Batch-size measurements on the frozen corpus were: batch 1, 13.13 s; batch 2, 13.64 s; batch 4, 21.01 s; batch 8, 25.91 s; and batch 16, 41.70 s. The corpus contains highly variable chunk lengths, so larger batches increased padding and peak activation cost on this CPU. The default is therefore 1, while `LOCAL_EMBEDDING_BATCH_SIZE` remains configurable for other hardware and corpora.

## Migration and cache safety

Production now defaults to `google/embeddinggemma-2` through Sentence Transformers. It uses `encode_query()` and `encode_document()` when available, falls back to the official `SearchQuery` and `Document` prompt names without manually prepending text, and explicitly normalizes every vector. The model is initialized text-only with vision and audio disabled. CPU uses float32; CUDA uses bfloat16 only when supported. Float16 is not used.

The index schema was incremented and the persisted index signature now includes provider, model, Sentence Transformers implementation, text-only modality selection, precision policy, prompt names, dimension, and profile version. Re-uploading a PDF after migration cannot reuse a BGE index or an older Gemma profile; it is parsed and embedded again, and obsolete vectors for that document are removed.

## Conclusion and limitations

EmbeddingGemma 2 improves Recall@1 by 2.9 points over the recalibrated BGE comparator and restores no-answer accuracy from the invalid preliminary 0% result to 100%. It does **not** outperform BGE overall on this dataset: Recall@5 is 2.9 points lower, Precision@5 is 0.6 points lower, MRR is effectively tied, and indexing remains materially slower and more memory intensive. The migration is implemented as requested, but the evidence supports a quality trade-off rather than a blanket accuracy win.

Remaining limitations are the small negative set, the nine-case holdout, absence of the original PDF for a fresh parse-to-index benchmark, exact rather than ANN search, and cross-encoder-dominated query latency. A larger domain-specific evaluation set should be collected before treating 0.630 as universal across documents.
