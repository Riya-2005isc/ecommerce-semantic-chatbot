from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from documents import documents, categories

app = FastAPI()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create document embeddings
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

# Normalize embeddings
faiss.normalize_L2(embeddings)

# Create FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "E-Commerce Semantic Search Chatbot API is running"
    }


@app.post("/chat")
def chat(request: QueryRequest):

    query = request.query

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, 3)

    top_index = indices[0][0]

    predicted_category = categories[top_index]
    response = documents[top_index]

    return {
        "query": query,
        "category": predicted_category,
        "response": response,
        "similarity_score": float(scores[0][0])
    }
