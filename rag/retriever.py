from sentence_transformers import SentenceTransformer
import pickle
import faiss

model=SentenceTransformer("all-MiniLM-L6-v2")
index=faiss.read_index("vector_store/resume.index")

with open ("vector_store/chunks.pk1","rb") as f:
    chunks=pickle.load(f)

query="Python,sql,machine learning,excel,django"

query_embedding=model.encode([query])

distance,indices=index.search(query_embedding,k=3)

print("The Top Matching Resumes are: \n")

for i in indices[0]:
    print("Resume : ", chunks[i])
    print(chunks[i].page_content)
    



