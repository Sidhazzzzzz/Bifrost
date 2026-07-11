import json
from app.classifier import classify

with open("tests/eval_dataset.json") as f:
    data = json.load(f)

for d in data:
    c = classify(d["prompt"])
    print(f"{c.complexity_score:.2f} {c.routed_to.value} - {d['category']}: {d['prompt'][:30]}")
