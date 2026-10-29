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
