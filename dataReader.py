import json

def readJSON(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    sentences = []

    for dictionary in data:
        sentences.append(dictionary["text"])

    return sentences



