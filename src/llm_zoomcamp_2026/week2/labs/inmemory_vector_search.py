"""Lab: Embed the FAQ dataset and explore vector search.

Week: 2 | Lesson: 3–5 | Type: Lab
"""

import numpy as np
from minsearch import VectorSearch
from sentence_transformers import SentenceTransformer

from llm_zoomcamp_2026.week1.core.ingest import load_documents


def main() -> None:
    """Compare manual dot-product ranking with minsearch results."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = load_documents()

    texts = [doc["question"] + " " + doc["answer"] for doc in documents]

    X = model.encode(
        texts,
        batch_size=50,
        show_progress_bar=True,
    )

    print(X.shape)

    print("Documents:", len(documents))
    print("Matrix shape:", X.shape)
    print("Ndim:", X.ndim)

    query = "Can I still join the course after the start date?"

    v_query = model.encode(query)
    # Dot products equal cosine similarity only for normalized vectors.
    scores = X.dot(v_query)
    best_index = np.argmax(scores)
    top_k_indices = np.argsort(scores)[-5:][::-1]

    print("Best index:", best_index)
    print("Best score:", scores[best_index])
    print("Best document:", documents[best_index])
    print("\nBest 5 documents by dot product:")
    for rank, index in enumerate(top_k_indices, start=1):
        print(f"{rank}. {documents[index]}")
        print(f"Score: {scores[index]}")

    print("\nminsearch results:")
    vindex = VectorSearch(keyword_fields=["course"])
    vindex.fit(X, documents)

    results = vindex.search(v_query, num_results=5)

    for rank, doc in enumerate(results, start=1):
        print(f"{rank}. {doc['question']}")
        print(f"Course: {doc['course']}")


if __name__ == "__main__":
    main()
