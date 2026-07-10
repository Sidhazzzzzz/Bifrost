import asyncio
from app.classifier import classify
from app.config import Settings
from app.client import LLMClient
from app.router import ModelRouter
from app.cache import PersistentResponseCache
from app.orchestrator import Orchestrator

async def main():
    settings = Settings()
    client = LLMClient(
        local_base_url=settings.local_base_url,
        remote_base_url=settings.remote_base_url,
        remote_api_key=settings.remote_api_key,
        remote_fallback_model=settings.remote_model
    )
    router = ModelRouter(settings)
    cache = PersistentResponseCache(
        settings.cache_path,
        settings,
        similarity_threshold=settings.cache_similarity_threshold,
        max_entries=settings.cache_max_entries,
    )
    orch = Orchestrator(settings, client, router, cache)
    
    prompt = '"Summarize this sentence: The quick brown fox jumps over the lazy dog." Submit, and highlight the UI showing it was routed locally'
    
    result = await orch.execute_task(prompt)
    print("Final Result:")
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
