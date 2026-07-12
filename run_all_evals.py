import json
import asyncio
import time
from app.orchestrator import Orchestrator
from app.config import Settings
from app.client import LLMClient
from app.router import ModelRouter
from app.cache import PersistentResponseCache
import os

async def main():
    # Remove responses cache to get raw token usage
    if os.path.exists('C:\\cache\\responses.json'):
        os.remove('C:\\cache\\responses.json')
        
    s = Settings()
    c = LLMClient(s)
    r = ModelRouter(s)
    cache = PersistentResponseCache(s)
    orch = Orchestrator(s, c, r, cache)
    
    datasets = ['very_easy', 'easy', 'moderate', 'hard', 'very_hard']
    results = {}
    
    for ds_name in datasets:
        with open(f'tests/gradingsets/{ds_name}.json', 'r') as f:
            tasks = json.load(f)
        
        correct = 0
        total_tokens = 0
        cache.exact_cache.clear()
        
        for task in tasks:
            resp = await orch.execute_task(task['prompt'])
            total_tokens += resp['total_tokens']
            
            got = str(resp.get('response', '')).lower()
            expected = str(task['ground_truth']).lower()
            
            # Extract expected tokens and check if they are all in the response
            # for multi-entity NER
            expected_items = [x.strip() for x in expected.split(',')]
            all_found = all(item in got for item in expected_items)
            
            if all_found:
                correct += 1
                
        results[ds_name] = {'correct': correct, 'total': len(tasks), 'tokens': total_tokens}
    
    print('\n=== EVALUATION REPORT ===')
    for ds_name, stats in results.items():
        acc = (stats['correct'] / stats['total']) * 100
        print(f'{ds_name.upper():<10} | Accuracy: {acc:5.1f}% ({stats["correct"]}/{stats["total"]}) | Tokens: {stats["tokens"]}')
    print('=========================')

asyncio.run(main())
