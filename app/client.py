"""
Bifrost API Client (LOCAL & REMOTE endpoints).

The remote endpoint is OpenAI-compatible and can be Fireworks for submission
or Groq for local development.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from app.classifier import RouteTarget


@dataclass
class LLMResponse:
    text: str
    model_used: str
    routed_to: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class TokenStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    local_calls: int = 0
    remote_calls: int = 0
    total_verification_retries: int = 0

    def record(self, response: LLMResponse) -> None:
        self.total_calls += 1
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        self.total_tokens += response.total_tokens
        self.total_latency_ms += response.latency_ms
        if response.routed_to == RouteTarget.LOCAL.value:
            self.local_calls += 1
        else:
            self.remote_calls += 1
        if response.error:
            self.total_errors += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "total_verification_retries": self.total_verification_retries,
            "local_calls": self.local_calls,
            "remote_calls": self.remote_calls,
            "avg_latency_ms": (
                round(self.total_latency_ms / self.total_calls, 1)
                if self.total_calls > 0 else 0.0
            ),
        }


class LLMClient:
    """Manages two separate HTTP client pools for LOCAL and REMOTE models."""

    def __init__(
        self,
        local_base_url: str,
        remote_base_url: str,
        remote_api_key: str,
        remote_fallback_model: str,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._local_base_url = local_base_url.rstrip("/")
        self._remote_base_url = remote_base_url.rstrip("/")
        self._remote_api_key = remote_api_key
        self._remote_fallback_model = remote_fallback_model
        self._timeout = timeout
        self._max_retries = max_retries
        self.stats = TokenStats()

        # Local client (Ollama)
        self._local_client = httpx.AsyncClient(
            base_url=self._local_base_url,
            headers={
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

        # Remote client (Groq)
        self._remote_client = httpx.AsyncClient(
            base_url=self._remote_base_url,
            headers={
                "Authorization": f"Bearer {self._remote_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        target: RouteTarget,
        model: str,
        *,
        max_tokens: int = 300,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Send a completion request to either the local Ollama or remote Groq endpoint."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        client = self._local_client if target == RouteTarget.LOCAL else self._remote_client
        timeout_to_use = httpx.Timeout(45.0) if target == RouteTarget.LOCAL else httpx.Timeout(self._timeout)
        retries_to_use = 0 if target == RouteTarget.LOCAL else self._max_retries
        
        last_error: str | None = None
        for attempt in range(retries_to_use + 1):
            start_time = time.perf_counter()
            try:
                resp = await client.post(
                    "/chat/completions",
                    json=payload,
                    timeout=timeout_to_use,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if resp.status_code == 429:
                    wait = min(2 ** attempt, 8)
                    await asyncio.sleep(wait)
                    last_error = "Rate limited"
                    continue

                resp.raise_for_status()
                data = resp.json()

                usage = data.get("usage", {})
                choices = data.get("choices", [])
                text = ""
                if choices:
                    text = choices[0].get("message", {}).get("content", "")

                # local models might not return exact tokens in usage, estimate if empty
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                if prompt_tokens == 0 and text:
                    # Rough token estimation
                    prompt_tokens = len(str(messages).split()) // 3
                    completion_tokens = len(text.split()) // 3

                result = LLMResponse(
                    text=text.strip(),
                    model_used=model,
                    routed_to=target.value,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=round(elapsed_ms, 1),
                )
                self.stats.record(result)
                return result

            except httpx.TimeoutException:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                last_error = f"Timeout after {elapsed_ms:.0f}ms"
                if attempt < self._max_retries:
                    await asyncio.sleep(1)

            except httpx.HTTPStatusError as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                last_error = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                if attempt < self._max_retries:
                    await asyncio.sleep(1)

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                break



        error_resp = LLMResponse(
            text="",
            model_used=model,
            routed_to=target.value,
            latency_ms=round(elapsed_ms, 1),
            error=last_error,
        )
        self.stats.record(error_resp)
        return error_resp

    async def get_embedding(
        self,
        text: str,
        target: RouteTarget = RouteTarget.LOCAL,
        model: str | None = None
    ) -> list[float]:
        """Fetch embedding for a given text."""
        # For local, default to mxbai-embed-large or all-minilm if model not specified.
        # Here we just use the same local model, or ollama's default.
        client = self._local_client if target == RouteTarget.LOCAL else self._remote_client
        payload = {
            "model": model or (self._remote_fallback_model if target == RouteTarget.REMOTE else "nomic-embed-text"),
            "input": text,
        }
        try:
            resp = await client.post(
                "/embeddings",
                json=payload,
                timeout=httpx.Timeout(10.0),
            )
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0].get("embedding", [])
            # Some local ollama endpoints might return 'embedding' directly if using /api/embeddings
            if "embedding" in data:
                return data["embedding"]
        except Exception as e:
            print(f"[Bifrost] Embedding fetch failed: {e}")
        return []

    async def close(self) -> None:
        await asyncio.gather(
            self._local_client.aclose(),
            self._remote_client.aclose(),
        )

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
