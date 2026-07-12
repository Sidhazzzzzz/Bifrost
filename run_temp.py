import asyncio
from app.config import load_settings
from app.processor import run_batch

import os

async def main():
    if os.path.exists('C:\\cache\\responses.json'):
        os.remove('C:\\cache\\responses.json')
    s = load_settings()
    s.input_path = 'tests/fresh_dataset.json'
    s.output_path = 'temp_results.json'
    
    import time
    start = time.perf_counter()
    await run_batch(s)
    end = time.perf_counter()
    print(f"Total batch time: {end - start:.2f}s")

asyncio.run(main())
