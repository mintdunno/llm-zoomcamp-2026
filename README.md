# LLM Zoomcamp 2026

Exercises and implementations from [DataTalksClub's LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp), with personal experiments and extensions. This is a learning repository: the code favors visible steps and experimentation over production architecture.

## Stack

Python, uv, Jupyter, OpenAI-compatible API, minsearch, Sentence Transformers, NumPy, sqlitesearch, SQLite, PostgreSQL, pgvector, psycopg, Docker Compose, and ToyAIKit.

## Organization and progress

Code is grouped by week under [`src/llm_zoomcamp_2026`](src/llm_zoomcamp_2026/):

| Folder | Purpose |
| --- | --- |
| `labs/` | Concept-focused exercises |
| `demos/` | End-to-end workflow examples |
| `core/` | Reusable implementations |
| `scripts/` | Data ingestion and operational tasks |

- **Week 1 — Agentic RAG:** implemented FAQ loading, keyword RAG, function calling, a manual agent loop, and a ToyAIKit example. The [notebook](notebooks/week1.ipynb) retains earlier work.
- **Week 2 — Vector Search:** implemented embedding experiments, in-memory vector search and RAG, SQLite ingestion and RAG, and pgvector ingestion and search. These are learning implementations; their presence does not imply all course exercises or homework are complete.
- **In progress / planned:** no specific additional work is recorded in the repository. The course's optional ONNX embedder has no implementation here.

## Learning maps

Explore the complete lesson-by-lesson maps for [Week 1 — Agentic RAG](learning_maps/week1_agentic_rag.md) and [Week 2 — Vector Search](learning_maps/week2_vector_search.md), including progress, source files, and commands. See the [map index](learning_maps/README.md) for status definitions.

## Setup

Install Python 3.14 or newer and [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
uv sync
```

Copy `.env.example` to `.env` and configure `API_KEY`, `BASE_URL`, `MODEL`, and `EMBEDDING_MODEL` for your OpenAI-compatible provider. `DATABASE_URL` selects PostgreSQL. Keep `.env` private; it is ignored by Git. The keyword RAG demo also accepts optional `INPUT_PRICE_PER_M`, `OUTPUT_PRICE_PER_M`, and `CACHED_INPUT_PRICE_PER_M` for cost estimates.

## Running examples

Run commands from the repository root. FAQ-loading examples access the network; local Sentence Transformers examples may download model weights on first use. API demos send requests to your configured provider and may incur charges.

```bash
# Keyword RAG and the manual agent (external LLM API)
uv run rag-demo
uv run agent-demo
uv run python -m llm_zoomcamp_2026.week1.demos.tool_calling
uv run python -m llm_zoomcamp_2026.week1.demos.framework_agent

# Local embeddings and in-memory vector search (model download; FAQ download for search)
uv run python -m llm_zoomcamp_2026.week2.labs.embedding_similarity
uv run python -m llm_zoomcamp_2026.week2.labs.inmemory_vector_search

# API embeddings and in-memory RAG (external API; RAG also needs the local model and FAQ)
uv run python -m llm_zoomcamp_2026.week2.labs.api_embeddings
uv run python -m llm_zoomcamp_2026.week2.demos.inmemory_vector_rag
```

The SQLite demo requires an existing `faq_vectors.db` in the current directory, the local model, and an LLM API. **Ingestion creates persistent data** and exits if that file already exists:

```bash
uv run python -m llm_zoomcamp_2026.week2.scripts.ingest_sqlite
uv run python -m llm_zoomcamp_2026.week2.demos.sqlite_vector_rag
```

## PostgreSQL and pgvector

The included Compose service runs PostgreSQL 17 with the pgvector extension at `127.0.0.1:5433`, using database `faq` and user `faq_user`. Its development connection URL is in `.env.example`. The SQL initialization script runs when the Compose data volume is first initialized; existing volumes are not reinitialized.

```bash
docker compose up -d
```

**Ingestion upserts persistent rows** in the configured database and downloads the FAQ and model if needed. Search requires the same embedding model and ingested data:

```bash
uv run python -m llm_zoomcamp_2026.week2.scripts.ingest_pgvector
uv run python -m llm_zoomcamp_2026.week2.demos.pgvector_search
```

The SQLite examples use an IVF index; the PostgreSQL query orders by cosine distance. These are learning examples and are not benchmarks of production retrieval quality.
