from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

q1 = "Can I still join the course after the start date?"

d = (
    "You don't need to register. You're accepted. "
    "You can also just start learning and submitting "
    "homework without registering."
)

q2 = "How to install Docker on Windows?"

v1 = model.encode(q1)
dv = model.encode(d)
v2 = model.encode(q2)

print("Vector shape:", v1.shape)
print("Q1 vs Document:", v1.dot(dv))
print("Q2 vs Document:", v2.dot(dv))
