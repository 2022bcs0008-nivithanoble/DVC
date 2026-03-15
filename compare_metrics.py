import json

with open("outputs/metrics/results.json") as f:
    metrics = json.load(f)

if metrics["accuracy"] > 0.0:
    print("Model improved")
else:
    raise Exception("Model performance not good")