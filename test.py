import json

from fastapi.testclient import TestClient
from main import app
import dataReader

client = TestClient(app)

def test_ingest():

    # Read the data.json file
    with open("data.json", "r") as file:
        data = json.load(file)

    response = client.post("/ingest", json=data)

    # Assert that the request was successful (HTTP 200 OK)
    assert response.status_code == 200

    # Assert all text was ingested correctly
    document = response.json()["ingested text"]

    assert data == document

def test_query():
    sentences = dataReader.readJSON("data.json")

    # Read the data.json file
    with open("data.json", "r") as file:
        data = json.load(file)

    client.post("/ingest", json=data)

    # Try the exact same sentence, which should return the corresponding sentence in the stored passage with distance 0
    # This is sentence with id:4
    response = client.get("/query",
                           params={"text": "Unlike traditional identity verification platforms, Civic's focus is on bridging the gap between users and organizations through intelligent communications pipelines."})

    # Assert that the request was successful (HTTP 200 OK)
    assert response.status_code == 200

    assert sentences[3] == response.json()["closest text"]
    assert 0 == response.json()["distance"]

    #Try other sentences
    response = client.get("/query",
                          params={
                              "text": "San Francisco is the best place to be. I love going there for vacations"})
    # Only sentence that menionts san francisco is id:6
    assert sentences[5] == response.json()["closest text"]

    response = client.get("/query",
                          params={
                              "text": "Can Civic be used as an automation tool to help me?"})

    assert sentences[0] == response.json()["closest text"]