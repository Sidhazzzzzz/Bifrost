import json
import asyncio
from app.router import run_pipeline

async def main():
    with open("tests/fresh_dataset.json") as f:
        data = json.load(f)
    
    sentiment_tasks = [t for t in data if t.get("category") == "SENTIMENT"]
    print(f"Found {len(sentiment_tasks)} SENTIMENT tasks.")
    
    results = {}
    for task in sentiment_tasks:
        print(f"\nTask ID: {task['id']}")
        print(f"Prompt: {task['prompt']}")
        import time
        start = time.time()
        res = await run_pipeline(task["prompt"])
        end = time.time()
        print(f"Result: {res}")
        print(f"Latency: {end - start:.2f}s")
        results[task["id"]] = res
        
if __name__ == "__main__":
    asyncio.run(main())
