import asyncio
import os
from httpx import AsyncClient
from app.client import LLMClient
from app.classifier import RouteTarget
from app.prompts import SYSTEM_PROMPTS, Category
from app.config import get_settings

async def main():
    settings = get_settings()
    async with AsyncClient() as http_client:
        client = LLMClient(
            http_client,
            remote_base_url="https://api.groq.com/openai/v1",
            remote_api_key=os.getenv("GROQ_API_KEY", "dummy"),
            remote_fallback_model="llama-3.1-8b-instant"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[Category.MATH]},
            {"role": "user", "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"}
        ]
        resp = await client.chat(
            messages=messages,
            target=RouteTarget.LOCAL,
            model="gemma2:2b",
            max_tokens=100,
            temperature=0.1
        )
        print("Response:", resp.text)
        print("Error:", resp.error)

if __name__ == "__main__":
    asyncio.run(main())
