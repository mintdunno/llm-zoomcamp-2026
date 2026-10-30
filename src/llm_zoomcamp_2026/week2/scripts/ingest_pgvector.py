"""Upsert FAQ documents and embeddings into PostgreSQL."""

import hashlib
import json
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from llm_zoomcamp_2026.week1.core.ingest import load_documents

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
BATCH_SIZE = 50

UPSERT_SQL = """
INSERT INTO documents (
    document_key,
    course,
    section,
    question,
    answer,
    embedding,
    embedding_model
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (document_key)
DO UPDATE SET
    course = EXCLUDED.course,
    section = EXCLUDED.section,
    question = EXCLUDED.question,
    answer = EXCLUDED.answer,
    embedding = EXCLUDED.embedding,
    embedding_model = EXCLUDED.embedding_model;
"""


def make_document_key(document: dict) -> str:
    """Generate a stable identifier for a FAQ document."""
    identity = json.dumps(
        [
            document["course"],
            document["section"],
            document["question"],
        ],
        ensure_ascii=False,
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main() -> None:
    """Generate embeddings and upsert FAQ documents into PostgreSQL."""
    load_dotenv()

    # 1. Load FAQ documents.
    documents = load_documents()

    if not documents:
        raise RuntimeError("No FAQ documents found.")

    # 2. Initialize embedding model.
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Prepare document texts.
    texts = [f"{doc['question']} {doc['answer']}" for doc in documents]

    # 4. Generate embeddings.
    vectors = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if vectors.shape != (
        len(documents),
        EMBEDDING_DIMENSIONS,
    ):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")

    # 5. Prepare records for database insertion.
    records = []

    for doc, vector in zip(documents, vectors, strict=True):
        records.append(
            (
                make_document_key(doc),
                doc["course"],
                doc["section"],
                doc["question"],
                doc["answer"],
                vector,
                EMBEDDING_MODEL,
            )
        )

    # 6. Connect to PostgreSQL and upsert records.
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        # Register adapters so psycopg can send NumPy vectors.
        register_vector(conn)

        with conn.cursor() as cursor:
            cursor.executemany(UPSERT_SQL, records)

        # Commit on success, rollback on exception.

    print(f"Successfully processed {len(records)} documents.")


if __name__ == "__main__":
    main()
