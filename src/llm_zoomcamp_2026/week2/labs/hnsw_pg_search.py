"""Compare exact and HNSW approximate vector search in PostgreSQL using pgvector."""

import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


def main() -> None:
    """Run vector search benchmarks comparing exact search with HNSW at different ef_search settings."""
    load_dotenv()

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    question = "Can I still join the course after it started?"
    course = "llm-zoomcamp"
    top_k = 5

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    v_question = embedder.encode(question)

    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as conn:
        register_vector(conn)
        with conn.cursor() as cursor:
            search_sql = """
                SELECT id
                FROM faq_hnsw_lab
                WHERE course = %s
                AND embedding_model = %s
                ORDER BY embedding <=> %s
                LIMIT %s
            """

            params = (course, EMBEDDING_MODEL, v_question, top_k)

            # 1. Exact Search: disable HNSW index scans.
            cursor.execute("SET LOCAL enable_indexscan = off")
            cursor.execute(search_sql, params)

            exact_ids = [row[0] for row in cursor.fetchall()]
            print("Exact Top-5:", exact_ids)

            # 2. HNSW Search: enable index scans.
            cursor.execute("SET LOCAL enable_indexscan = on")
            cursor.execute("SET LOCAL enable_seqscan = off")

            for ef in (10, 40, 100):
                cursor.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true)",
                    (str(ef),),
                )

                cursor.execute(search_sql, params)
                hnsw_ids = [row[0] for row in cursor.fetchall()]

                recall = recall_at_k(exact_ids, hnsw_ids)

                print(f"\nef_search={ef}")
                print("HNSW IDs:", hnsw_ids)
                print(f"Recall@{top_k}: {recall:.0%}")


def recall_at_k(
    exact_ids: list[int],
    approx_ids: list[int],
) -> float:
    """Calculate Recall@K by comparing approximate result IDs against ground truth exact IDs."""
    if not exact_ids:
        return 0.0

    matches = set(exact_ids) & set(approx_ids)
    return len(matches) / len(exact_ids)


if __name__ == "__main__":
    main()
