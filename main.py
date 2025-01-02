from sentence_transformers import SentenceTransformer

import dataReader
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np



model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = dataReader.readJSON("data.json")

embeddings = model.encode(sentences)
print(embeddings.shape)
vector_dimension = embeddings.shape[1]

# Create a FAISS Index with correct dimensions
index = faiss.IndexFlatL2(vector_dimension)

app = FastAPI()

texts = []

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ingest")
def ingest(text):
    # Generate embedding for the input text
    embedding = model.encode([text])

    # Add embedding to FAISS index
    index.add(embedding)
    texts.append(text)

    return {"text": text}


@app.get("/query")
def query(text: str):
    # Generate embedding for the query
    query_embedding = model.encode([text])

    # Perform a similarity search in the FAISS index
    distances, indices = index.search(query_embedding, 1)

    # Retrieve the most similar sentence
    closest_text = texts[indices[0][0]]


    return {"closest text": closest_text, "distance": distances[0][0]}


for s in sentences:
    ingest(s)



ingest("this is an example sentence")
print(texts)
query("this is an example sentence")