import asyncio
import os
from app.config import load_settings
from app.orchestrator import Orchestrator
from app.client import LLMClient
from app.router import ModelRouter
from app.cache import PersistentResponseCache

async def main():
    if os.path.exists('C:\\cache\\responses.json'):
        os.remove('C:\\cache\\responses.json')
    settings = load_settings()
    client = LLMClient(
        local_base_url=settings.local_base_url,
        remote_base_url=settings.remote_base_url,
        remote_api_key=settings.remote_api_key,
        remote_fallback_model=settings.remote_model,
        timeout=30.0,
    )
    router = ModelRouter(settings)
    cache = PersistentResponseCache(
        settings.cache_path,
        settings,
        similarity_threshold=settings.cache_similarity_threshold,
        max_entries=settings.cache_max_entries,
    )
    orchestrator = Orchestrator(settings, client, router, cache)
    print("Running task...")
    res = await orchestrator.execute_task('test_1', 'Extract the entities from this text: The United Nations Security Council met in the Middle East to discuss the World Health Organization.', 'ner')
    print(f'\nSanity check output: {res}')

asyncio.run(main())
