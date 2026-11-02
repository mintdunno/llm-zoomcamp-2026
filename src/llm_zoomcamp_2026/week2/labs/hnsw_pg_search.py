"""Benchmark Exact Search and HNSW search quality and latency."""

import os
import time
from statistics import mean, median

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


def recall_at_k(
    exact_ids: list[int],
    approx_ids: list[int],
) -> float:
    """Return the fraction of exact neighbors found by approximate search."""
    if not exact_ids:
        return 0.0

    matches = set(exact_ids) & set(approx_ids)
    return len(matches) / len(exact_ids)


def main() -> None:
    """Compare exact search and HNSW iterative scan configurations."""
    load_dotenv()

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    questions = [
        "Can I join the course after it has already started?",
        "How do I register for the LLM Zoomcamp course?",
        "Is the course completely free?",
        "Can I get a certificate after completing the course?",
        "What should I do if I miss a homework deadline?",
        "How do I submit my homework assignments?",
        "Can I submit homework after the deadline?",
        "How are homework assignments graded?",
        "Do I need a powerful GPU to complete the course?",
        "What Python knowledge is required before starting?",
        "Do I need prior experience with machine learning?",
        "How do I set up the development environment?",
        "Can I use Windows instead of Linux?",
        "How do I run PostgreSQL using Docker?",
        "What should I do if Docker is not working?",
        "Can I use a different LLM provider for the homework?",
        "How can I reduce the cost of using an LLM API?",
        "Where can I find the course materials and recordings?",
        "How do I ask questions or get help from the community?",
        "Can I complete the course at my own pace?",
    ]

    course = "llm-zoomcamp"
    top_k = 5

    modes = ("off", "strict_order", "relaxed_order")
    ef_values = (10, 40)

    # Benchmark metrics.
    exact_latencies = []

    recalls = {(mode, ef): [] for mode in modes for ef in ef_values}

    latencies = {(mode, ef): [] for mode in modes for ef in ef_values}

    returned_counts = {(mode, ef): [] for mode in modes for ef in ef_values}

    # 1. Encode all questions once.
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    v_questions = embedder.encode(questions)

    search_sql = """
        SELECT id
        FROM faq_hnsw_lab
        WHERE course = %s
          AND embedding_model = %s
        ORDER BY embedding <=> %s
        LIMIT %s
    """

    # 2. Connect to PostgreSQL.
    with psycopg.connect(
        os.environ["DATABASE_URL"],
        connect_timeout=10,
    ) as conn:
        register_vector(conn)

        for idx, (question, v_question) in enumerate(
            zip(questions, v_questions, strict=True),
            start=1,
        ):
            params = (
                course,
                EMBEDDING_MODEL,
                v_question,
                top_k,
            )

            with conn.cursor() as cursor:
                print(f"\nQuestion {idx}: {question}")

                # 3. Exact Search (ground truth).
                cursor.execute("SET LOCAL enable_seqscan = on")
                cursor.execute("SET LOCAL enable_indexscan = off")

                start = time.perf_counter()

                cursor.execute(
                    search_sql,
                    params,
                    prepare=False,
                )

                exact_ids = [row[0] for row in cursor.fetchall()]

                elapsed_ms = (time.perf_counter() - start) * 1000
                exact_latencies.append(elapsed_ms)

                print(f"Exact Top-{top_k}: {exact_ids}")
                print(f"Exact latency: {elapsed_ms:.3f} ms")

                # 4. HNSW Approximate Search.
                cursor.execute("SET LOCAL enable_indexscan = on")
                cursor.execute("SET LOCAL enable_seqscan = off")

                for mode in modes:
                    cursor.execute(
                        "SELECT set_config('hnsw.iterative_scan', %s, true)",
                        (mode,),
                    )

                    for ef in ef_values:
                        cursor.execute(
                            "SELECT set_config('hnsw.ef_search', %s, true)",
                            (str(ef),),
                        )

                        start = time.perf_counter()

                        cursor.execute(
                            search_sql,
                            params,
                            prepare=False,
                        )

                        hnsw_ids = [row[0] for row in cursor.fetchall()]

                        elapsed_ms = (time.perf_counter() - start) * 1000

                        # Record latency.
                        latencies[(mode, ef)].append(elapsed_ms)

                        # Record recall.
                        recall = recall_at_k(
                            exact_ids,
                            hnsw_ids,
                        )

                        recalls[(mode, ef)].append(recall)

                        # Record returned document count.
                        returned_counts[(mode, ef)].append(len(hnsw_ids))

                        print(
                            f"  {mode:14} | "
                            f"ef={ef:3} | "
                            f"Recall@{top_k}={recall:.0%} | "
                            f"Returned={len(hnsw_ids)}/{top_k} | "
                            f"Latency={elapsed_ms:.3f} ms"
                        )

    # 5. Exact Search Summary.
    print("\n--- Exact Search Latency ---")
    print(f"Mean:   {mean(exact_latencies):.3f} ms")
    print(f"Median: {median(exact_latencies):.3f} ms")

    # 6. HNSW Recall Summary.
    print("\n--- HNSW Recall Benchmark ---")

    for mode in modes:
        for ef in ef_values:
            key = (mode, ef)

            avg_recall = mean(recalls[key])
            avg_returned = mean(returned_counts[key])

            print(
                f"{mode:14} | "
                f"ef={ef:3} | "
                f"Recall@{top_k}={avg_recall:.2%} | "
                f"Avg Returned={avg_returned:.2f}/{top_k}"
            )

    # 7. HNSW Latency Summary.
    print("\n--- HNSW Latency Benchmark ---")

    for mode in modes:
        for ef in ef_values:
            values = latencies[(mode, ef)]

            print(
                f"{mode:14} | "
                f"ef={ef:3} | "
                f"Mean={mean(values):.3f} ms | "
                f"Median={median(values):.3f} ms"
            )

    # 8. Exact vs HNSW Comparison.
    print("\n--- Exact vs HNSW Comparison ---")

    print(
        f"{'Exact Search':14} | "
        f"Recall@{top_k}=100.00% | "
        f"Median={median(exact_latencies):.3f} ms"
    )

    for mode in modes:
        for ef in ef_values:
            key = (mode, ef)

            print(
                f"{mode:14} | "
                f"ef={ef:3} | "
                f"Recall@{top_k}={mean(recalls[key]):.2%} | "
                f"Median={median(latencies[key]):.3f} ms"
            )


if __name__ == "__main__":
    main()
