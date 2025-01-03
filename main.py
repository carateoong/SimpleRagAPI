from typing import List

from sentence_transformers import SentenceTransformer

import dataReader
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = dataReader.readJSON("data.json")

embeddings = model.encode(sentences)
print(embeddings.shape)
vector_dimension = embeddings.shape[1]

# Create a FAISS Index with correct dimensions
index = faiss.IndexFlatL2(vector_dimension)

texts = []
class Sentence(BaseModel):
    id: int
    text: str


@app.get("/")
def read_root():
    return {"Hello": "perasdasdsio"}


@app.post("/ingest")
def ingest(document: List[Sentence]):

    for sentence in document:
        # Generate embedding for the input text
        embedding = model.encode([sentence.text])

        # Add embedding to FAISS index
        index.add(embedding)
        texts.append(sentence.text)


    print(texts)
    return {"ingested text": document}


@app.get("/query")
def query(text: str):
    # Generate embedding for the query
    query_embedding = model.encode([text])

    # Perform a similarity search in the FAISS index
    distances, indices = index.search(query_embedding, 1)

    # Retrieve the most similar sentence
    closest_text = texts[indices[0][0]]


    return {"closest text": closest_text, "distance": distances[0][0]}

