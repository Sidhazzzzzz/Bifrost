import json
import os
import sys
import time
import asyncio

from app.config import load_settings
from app.processor import run_batch

async def main():
    # 1. Clear old state
    if os.path.exists('router_stats.json'):
        os.remove('router_stats.json')
    if os.path.exists('app/stats.json'):
        os.remove('app/stats.json')
    if os.path.exists('C:\\cache\\responses.json'):
        os.remove('C:\\cache\\responses.json')
    
    with open('tests/fresh_dataset.json', 'r') as f:
        tasks = json.load(f)
        
    sentiment_tasks = [t for t in tasks if t.get('category').upper() == 'SENTIMENT']
    print(f"Found {len(sentiment_tasks)} SENTIMENT tasks.")
    
    with open('tests/sentiment_dataset.json', 'w') as f:
        json.dump(sentiment_tasks, f)
    
    os.environ["INPUT_PATH"] = "tests/sentiment_dataset.json"
    os.environ["OUTPUT_PATH"] = "output/sentiment_results.json"
    settings = load_settings()
    
    start_time = time.time()
    await run_batch(settings)
    end_time = time.time()
    
    wall_clock_ms = (end_time - start_time) * 1000
    print(f"\nBATCH WALL-CLOCK TIME: {wall_clock_ms:.1f}ms")

    with open('output/sentiment_results.json', 'r') as f:
        results_data = json.load(f)
        
    results_map = {r['task_id']: r for r in results_data}
    
    print("\n--- Accuracy Report ---")
    correct = 0
    total = len(sentiment_tasks)
    for task in sentiment_tasks:
        task_id = task['task_id']
        expected = str(task.get('ground_truth', '')).lower()
        got_resp = results_map.get(task_id, {})
        got = str(got_resp.get('answer', '')).lower()
        latency = got_resp.get('latency_ms', 0)
        
        expected_items = [x.strip() for x in expected.split(',')]
        all_found = all(item in got for item in expected_items)
        
        if all_found:
            correct += 1
            print(f"PASS [{task_id}] - Latency: {latency:.1f}ms")
        else:
            print(f"FAILED [{task_id}] - Latency: {latency:.1f}ms - Expected '{expected}', Got: '{got}'")
            
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\nFINAL ACCURACY: {acc:.1f}% ({correct}/{total})")

if __name__ == "__main__":
    asyncio.run(main())
