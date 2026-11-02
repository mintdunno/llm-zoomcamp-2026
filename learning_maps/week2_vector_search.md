---

## Extension — PostgreSQL HNSW Indexing & Evaluation

**Status:** Completed (learning lab)

This extension explores approximate nearest-neighbor search
using PostgreSQL, pgvector, and HNSW.

### Learning objectives

- Understand HNSW graph construction and multilayer search.
- Compare Exact Search with approximate HNSW Search.
- Understand `m`, `ef_construction`, and `ef_search`.
- Analyze execution plans with `EXPLAIN (ANALYZE, BUFFERS)`.
- Evaluate retrieval accuracy using Recall@K.
- Understand post-filtering in ANN search.
- Compare iterative scan modes.
- Build and evaluate a Partial HNSW Index.
- Measure query latency with Python.

### Relevant files

| File                                                                                           | Purpose                                             |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| [pgvector_hnsw_lab_ingest.py](../src/llm_zoomcamp_2026/week2/labs/pgvector_hnsw_lab_ingest.py) | Generate embeddings and ingest FAQ documents        |
| [hnsw_pg_search.py](../src/llm_zoomcamp_2026/week2/labs/hnsw_pg_search.py)                     | Benchmark Exact Search, HNSW, Recall@5, and latency |
| [faq_hnsw_lab.sql](../sql/labs/faq_hnsw_lab.sql)                                               | Define the lab table and HNSW indexes               |

### Environment

- PostgreSQL 17 with pgvector
- Python 3.14
- Psycopg3
- SentenceTransformers
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector dimensions: 384
- Distance metric: Cosine distance (`<=>`)

### Run the lab

Start PostgreSQL:

```bash
docker compose up -d
```

On a fresh database, apply `sql/labs/faq_hnsw_lab.sql`
before ingestion.

Ingest FAQ documents:

```bash
uv run python -m llm_zoomcamp_2026.week2.labs.pgvector_hnsw_lab_ingest
```

Run the HNSW benchmark:

```bash
uv run python -m llm_zoomcamp_2026.week2.labs.hnsw_pg_search
```

### Dataset and evaluation setup

| Property               | Value |
| ---------------------- | ----- |
| Total documents        | 1,457 |
| Matching documents     | 161   |
| Evaluation queries     | 20    |
| Top-K                  | 5     |
| Embedding dimensions   | 384   |
| HNSW `m`               | 16    |
| HNSW `ef_construction` | 64    |

Recall@5 uses Exact Search as the nearest-neighbor
ground truth and compares document IDs.

### Experiment 1 — Full HNSW Index

The HNSW index covers all documents, while queries filter
by course and embedding model.

| ef_search | Average Recall@5 |
| --------- | ---------------- |
| 10        | 34%              |
| 20        | 53%              |
| 40        | 80%              |
| 100       | 96%              |

**Finding:** Increasing `ef_search` improves recall,
but higher search breadth can increase query latency.

Post-filtering can also cause HNSW to return fewer than
the requested K results.

### Experiment 2 — Iterative HNSW Scan

Compared three modes:

- `off`: No automatic continued index scan.
- `strict_order`: Continue scanning while preserving distance order.
- `relaxed_order`: Continue scanning with relaxed distance ordering.

Full HNSW results:

| Mode          | ef_search | Recall@5 | Avg Returned |
| ------------- | --------- | -------- | ------------ |
| off           | 10        | 34%      | 1.70/5       |
| off           | 40        | 80%      | 4.00/5       |
| strict_order  | 10        | 100%     | 5.00/5       |
| strict_order  | 40        | 100%     | 5.00/5       |
| relaxed_order | 10        | 100%     | 5.00/5       |
| relaxed_order | 40        | 100%     | 5.00/5       |

**Finding:** Iterative scans improved recall when the
course filter removed many HNSW candidates.

### Experiment 3 — Partial HNSW Index

Created an HNSW index limited to:

- Course: `llm-zoomcamp`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

Verified index usage with:

```sql
EXPLAIN (ANALYZE, BUFFERS)
```

PostgreSQL selected:

`ix_faq_hnsw_lab_llm_partial`

Results:

| Mode          | ef_search | Recall@5 |
| ------------- | --------- | -------- |
| off           | 10        | 99%      |
| off           | 40        | 100%     |
| strict_order  | 10        | 99%      |
| strict_order  | 40        | 100%     |
| relaxed_order | 10        | 99%      |
| relaxed_order | 40        | 100%     |

**Finding:** Partial indexing substantially improved
filtered ANN retrieval without requiring iterative scans.

### Experiment 4 — Exact vs Partial HNSW Latency

Measured query latency using `time.perf_counter()`,
including query execution and result fetching.

| Method              | Recall@5 | Median latency |
| ------------------- | -------- | -------------- |
| Exact Search        | 100%     | 1.421 ms       |
| HNSW off, ef=10     | 99%      | 0.817 ms       |
| HNSW off, ef=40     | 100%     | 0.893 ms       |
| HNSW strict, ef=10  | 99%      | 0.715 ms       |
| HNSW strict, ef=40  | 100%     | 0.732 ms       |
| HNSW relaxed, ef=10 | 99%      | 0.824 ms       |
| HNSW relaxed, ef=40 | 100%     | 0.958 ms       |

**Finding:** Partial HNSW achieved competitive latency
and high recall in this small-scale experiment.

### Important engineering lessons

1. Recall@K evaluates approximate retrieval against exact
   nearest neighbors, not semantic relevance.
2. High `ef_search` can improve recall at additional cost.
3. Post-filtering may produce fewer than K results.
4. Iterative scans can compensate for filtering losses.
5. Partial HNSW indexes reduce the candidate search space.
6. Partial indexes introduce additional write and storage costs.
7. Prepared statements and query planning can affect
   benchmark validity.
8. Index selection must be verified using `EXPLAIN`.

### Benchmark limitations

These results are preliminary:

- Only 20 evaluation queries.
- Small dataset with 161 matching documents.
- One timing sample per query per configuration.
- No dedicated warm-up or randomized execution order.
- Python timing includes driver and communication overhead.
- Exact Search was measured with index scans disabled.
- Results do not establish production performance.

The observed 100% Recall@5 applies only to the tested
queries and configurations.

### Conclusion

HNSW is useful for approximate vector retrieval, but its
effectiveness depends on index design, query filtering,
search parameters, and workload.

For this dataset, Partial HNSW significantly improved recall
compared with Full HNSW without iterative scanning.

Further optimization would require larger datasets,
controlled benchmarks, and production-like workloads.
