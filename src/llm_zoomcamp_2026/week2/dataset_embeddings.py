import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from llm_zoomcamp_2026.week1.ingest import load_documents

model = SentenceTransformer("all-MiniLM-L6-v2")
documents = load_documents()

texts = [doc["question"] + " " + doc["answer"] for doc in documents]

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i : i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

X = np.array(vectors)

print("Documents:", len(documents))
print("Matrix shape:", X.shape)
print("Ndim:", X.ndim)

query = "Can I still join the course after the start date?"

v_query = model.encode(query)
scores = X.dot(v_query)

idx = np.argmax(scores)
best_5 = np.argsort(scores)[-5:][::-1]

print("Best index:", idx)
print("Best score:", scores[idx])
print("Best document:", documents[idx])

print("\nBest 5 documents:")

ordinals = ["1st", "2nd", "3rd", "4th", "5th"]

for rank, i in enumerate(best_5):
    print(f"{ordinals[rank]} document: {documents[i]}")
    print(f"Score: {scores[i]}")
