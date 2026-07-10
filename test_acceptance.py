import json
from app.classifier import classify, Category
from app.quality import is_weak_answer

print("--- sample_tasks.json ---")
with open("tests/sample_tasks.json") as f:
    tasks = json.load(f)
for t in tasks:
    c = classify(t["prompt"])
    print(f"Task {t['task_id']}: Expected {t.get('category', 'N/A')}, Got {c.category.value}")

print("\n--- Negative Code Checks ---")
negatives = [
    "What do I get in return for my investment?",
    "What is the tax return deadline this year?",
    "What class of vertebrate is a shark?",
    "The store has a 30-day return policy, is that generous?",
    "Summarize the following: ... the development of algorithms ..."
]
for n in negatives:
    c = classify(n)
    print(f"Got {c.category.value} for: {n[:40]}...")

print("\n--- Positive Code Debug Checks ---")
positives = [
    "Why does my code crash with a segfault?",
    "My recursive fibonacci function returns the wrong value for n=10, why?"
]
for p in positives:
    c = classify(p)
    print(f"Got {c.category.value} for: {p[:40]}...")

print("\n--- Logic Check ---")
logic = "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?"
c = classify(logic)
print(f"Got {c.category.value} for Logic Check")

print("\n--- Weak Answer Check ---")
is_weak = is_weak_answer("fix this", "def add(a, b):\n    return a + b", Category.CODE_DEBUG)
print(f"is_weak_answer returned: {is_weak}")
