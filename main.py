from typing import List

from sentence_transformers import SentenceTransformer

import dataReader
import faiss
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = dataReader.readJSON("data.json")

embeddings = model.encode(sentences)
vector_dimension = embeddings.shape[1]

# Create a FAISS Index with correct dimensions
index = faiss.IndexFlatL2(vector_dimension)

texts = []


#Create pydantic models to define fields and attributes
class Sentence(BaseModel):
    id: int
    text: str

class Input(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"safasf": "Hello, World!"}


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
def query(input: Input):
    print("texts")
    print(texts)
    print("input text:")
    print(input.text)
    # Generate embedding for the query
    query_embedding = model.encode([input.text])
    print("query embedding")
    print(query_embedding)
    print(query_embedding.shape)

    distances, indices = index.search(query_embedding, 1)

    # Retrieve the most similar sentence
    closest_text = texts[indices[0][0]]

    return {"closest text": closest_text, "distance": float(distances[0][0])}

