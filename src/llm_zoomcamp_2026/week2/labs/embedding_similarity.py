"""Lab: Compare text embeddings with dot products."""

from sentence_transformers import SentenceTransformer


def main() -> None:
    """Run this example."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    q1 = "Can I still join the course after the start date?"
    q2 = "How to install Docker on Windows?"

    d = (
        "You don't need to register. You're accepted. "
        "You can also just start learning and submitting "
        "homework without registering."
    )

    v1 = model.encode(q1)
    v2 = model.encode(q2)
    dv = model.encode(d)

    print("Shape:", dv.shape)
    # Dot product is cosine similarity only if both vectors are normalized.
    print("Q1 similarity:", v1.dot(dv))
    print("Q2 similarity:", v2.dot(dv))


if __name__ == "__main__":
    main()
