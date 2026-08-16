import streamlit as st
import os
import faiss
import pickle

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector_store/resume.index")

# Load stored chunks
with open("vector_store/chunks.pk1", "rb") as f:
    chunks = pickle.load(f)


def generate_answer(query):
    # Convert query into embedding
    query_embedding = model.encode([query]).astype("float32")

    # Retrieve top 3 matching resume chunks
    distances, indices = index.search(query_embedding, k=3)

    # Build context
    context = ""

    for i in indices[0]:
        context += chunks[i].page_content
        context += "\n\n"

    # Prompt
    prompt = f"""
You are an HR Recruitment AI Assistant.

Use ONLY the information from the resumes below to answer the question.

Resume Information:
{context}

Question:
{query}

Answer:
"""

    # Groq API Call
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


st.title("HR Resume ATS Screening ChatBot")

query=st.text_input("Ask any Question: ")
if st.button("Generate Answers"):
    answer=generate_answer(query)
    st.success(answer)
