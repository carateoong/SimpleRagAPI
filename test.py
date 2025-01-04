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


