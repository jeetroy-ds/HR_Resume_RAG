from sentence_transformers import SentenceTransformer
from utils.pdf_loader import load_resumes
from utils.text_splitter import split_documents

import faiss
import numpy as np
import pickle

def create_embeddings():
    documents=load_resumes("data/resumes")
    chunks=split_documents(documents)
    model=SentenceTransformer("all-MiniLM-L6-v2")

    text=[]
    for c in chunks:
        text.append(c.page_content)
    embeddings=model.encode(text)
    index=faiss.IndexFlatL2(384)
    embeddings=np.array(embeddings).astype("float32")
    index.add(embeddings)

    faiss.write_index(index,"vector_store/resume.index")
    with open("vector_store/chunks.pk1","wb") as f:
        pickle.dump(chunks,f)

    print(f"The Length of chunks is : {len(text)}")
    print(f"The Shape of embeddings is : {embeddings.shape}")
    print(f"The Total number of index are : {index.ntotal}")


if __name__ == "__main__":
    create_embeddings()