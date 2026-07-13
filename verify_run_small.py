import json
import time
import asyncio
import os
import sys

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
    
    os.environ["INPUT_PATH"] = "tests/small_dataset.json"
    os.environ["OUTPUT_PATH"] = "output/results.json"
    settings = load_settings()
    
    # 2. Run Batch with Timing
    start_time = time.time()
    await run_batch(settings)
    end_time = time.time()
    
    wall_clock_ms = (end_time - start_time) * 1000
    print(f"\n==========================================")
    print(f"BATCH WALL-CLOCK TIME: {wall_clock_ms:.1f}ms")
    print(f"==========================================")

    # 3. Calculate Accuracy
    with open('tests/small_dataset.json', 'r') as f:
        tasks = json.load(f)
        
    with open('output/results.json', 'r') as f:
        results_data = json.load(f)
        
    results_map = {r['task_id']: r for r in results_data}
    
    correct = 0
    total = len(tasks)
    
    print("\n--- Accuracy Report ---")
    for task in tasks:
        task_id = task['task_id']
        expected = str(task.get('ground_truth', '')).lower()
        got_resp = results_map.get(task_id, {})
        got = str(got_resp.get('answer', '')).lower()
        
        expected_items = [x.strip() for x in expected.split(',')]
        all_found = all(item in got for item in expected_items)
        
        if all_found:
            correct += 1
        else:
            print(f"FAILED [{task['category']}]: Expected '{expected}', Got: '{got}'")
            
    acc = (correct / total) * 100
    print(f"\nFINAL ACCURACY: {acc:.1f}% ({correct}/{total})")
    
if __name__ == "__main__":
    asyncio.run(main())
