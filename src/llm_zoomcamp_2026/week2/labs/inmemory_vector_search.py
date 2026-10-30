"""Lab: Embed the FAQ dataset and explore vector search."""

from minsearch import VectorSearch
from sentence_transformers import SentenceTransformer

from llm_zoomcamp_2026.week1.core.ingest import load_documents


def main() -> None:
    """Run this example."""
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
    # import numpy as np
    # scores = X.dot(v_query)

    # idx = np.argmax(scores)
    # best_5 = np.argsort(scores)[-5:][::-1]

    # print("Best index:", idx)
    # print("Best score:", scores[idx])
    # print("Best document:", documents[idx])

    # print("\nBest 5 documents:")

    # ordinals = ["1st", "2nd", "3rd", "4th", "5th"]

    # for rank, i in enumerate(best_5):
    #     print(f"{ordinals[rank]} document: {documents[i]}")
    #     print(f"Score: {scores[i]}")

    vindex = VectorSearch(keyword_fields=["course"])
    vindex.fit(X, documents)

    results = vindex.search(v_query, num_results=5)

    for rank, doc in enumerate(results, start=1):
        print(f"{rank}. {doc['question']}")
        print(f"Course: {doc['course']}")


if __name__ == "__main__":
    main()
