import httpx
import asyncio

async def test():
    payload = {
        "model": "gemma2:2b",
        "messages": [
            {"role": "system", "content": "You are a sentiment classifier. Output exactly ONE word: 'positive', 'negative', or 'neutral'. Do NOT output any other text, reasoning, or explanation. Note: reviews that express disappointment, lack of expected quality, or subtle criticism should be classified as 'negative'."},
            {"role": "user", "content": "The food was okay, but the service was terrible."}
        ],
        "max_tokens": 5,
        "temperature": 0.1
    }
    
    async with httpx.AsyncClient() as client:
        import time
        t0 = time.time()
        resp = await client.post("http://localhost:11434/v1/chat/completions", json=payload, timeout=60)
        t1 = time.time()
        print(f"Time: {t1 - t0:.2f}s")
        print(resp.json())

if __name__ == "__main__":
    asyncio.run(test())
