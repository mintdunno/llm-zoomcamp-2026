from sentence_transformers import SentenceTransformer

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
print("Q1 similarity:", v1.dot(dv))
print("Q2 similarity:", v2.dot(dv))
