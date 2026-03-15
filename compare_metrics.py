import json

with open("outputs/metrics/results.json") as f:
    metrics = json.load(f)

if metrics["R2"] > 0.0 or metrics["MSE"] < 4e9:
    print("Model improved")
else:
    raise Exception("Model performance not good")