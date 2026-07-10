"""SHA256 response cache helpers for API and batch routing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import math
import aiofiles
from app.config import Settings

CACHE_VERSION = "bifrost-cache-v2"

def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt)).strip().lower()


def prompt_tokens(prompt: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_prompt(prompt)))


def make_cache_key(
    prompt: str,
    settings: Settings,
    *,
    force_target: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "version": CACHE_VERSION,
        "prompt": normalize_prompt(prompt),
        "force_target": (force_target or "").upper(),
        "local_model": settings.local_model,
        "remote_model": settings.remote_model,
        "remote_provider": settings.remote_provider,
        "allowed_models": list(settings.allowed_models),
        "threshold": settings.complexity_threshold,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def route_signature(settings: Settings) -> str:
    payload = {
        "version": CACHE_VERSION,
        "local_model": settings.local_model,
        "remote_model": settings.remote_model,
        "remote_provider": settings.remote_provider,
        "allowed_models": list(settings.allowed_models),
        "threshold": settings.complexity_threshold,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class PersistentResponseCache:
    """Tiny JSON-backed cache.

    Exact matches are safe for every successful answer. Similar matches are
    intentionally restricted to zero-token deterministic answers to avoid
    returning a stale generated answer for a meaningfully different prompt.
    """

    def __init__(
        self,
        path: str,
        settings: Settings,
        *,
        similarity_threshold: float = 0.92,
        max_entries: int = 2000,
    ) -> None:
        self.path = Path(path)
        self.signature = route_signature(settings)
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.similar_hits = 0
        self._lock = asyncio.Lock()
        self._load()

    def get_exact(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if not entry or entry.get("signature") != self.signature:
            self.misses += 1
            return None
        self.hits += 1
        return dict(entry["response"])

    def get_similar(
        self,
        prompt: str,
        category: str,
        *,
        force_target: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> dict[str, Any] | None:
        if force_target:
            self.misses += 1
            return None

        # Exact token fast-path check
        tokens = prompt_tokens(prompt)
        
        # Fallback to Jaccard overlap

        best_score = 0.0
        best_response: dict[str, Any] | None = None
        for entry in self._entries.values():
            if entry.get("signature") != self.signature:
                continue
            if entry.get("category") != category:
                continue
            response = entry.get("response", {})
            if not self._safe_for_similar_reuse(response):
                continue
                
            cached_tokens = set(entry.get("tokens", []))
            if tokens and cached_tokens:
                score = len(tokens & cached_tokens) / len(tokens | cached_tokens)

            if score > best_score:
                best_score = score
                best_response = dict(response)

        if best_response and best_score >= self.similarity_threshold:
            self.hits += 1
            self.similar_hits += 1
            return best_response

        self.misses += 1
        return None

    def set(
        self,
        key: str,
        prompt: str,
        category: str,
        response: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> None:
        if response.get("error"):
            return
        self._entries[key] = {
            "signature": self.signature,
            "prompt": normalize_prompt(prompt),
            "tokens": sorted(prompt_tokens(prompt)),
            "category": category,
            "embedding": embedding or [],
            "response": response,
        }
        self._trim()
        self._save()

    async def set_async(
        self,
        key: str,
        prompt: str,
        category: str,
        response: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> None:
        if response.get("error"):
            return
            
        async with self._lock:
            self._entries[key] = {
                "signature": self.signature,
                "prompt": normalize_prompt(prompt),
                "tokens": sorted(list(prompt_tokens(prompt))),
                "category": category,
                "embedding": embedding or [],
                "response": response,
            }
            self._trim()
            await self._save_async()

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "similar_hits": self.similar_hits,
        }

    @staticmethod
    def _safe_for_similar_reuse(response: dict[str, Any]) -> bool:
        cat = response.get("category", "")
        if cat in {"code_generation", "code_debugging", "code_gen", "code_debug"}:
            return False
        if response.get("error"):
            return False
        return True

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._entries = {
                        str(key): value
                        for key, value in raw.get("entries", {}).items()
                        if isinstance(value, dict)
                    }
        except Exception:
            self._entries = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_VERSION, "entries": self._entries}
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    async def _save_async(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_VERSION, "entries": self._entries}
        async with aiofiles.open(self.path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(payload, indent=2, ensure_ascii=False))

    def _trim(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        overflow = len(self._entries) - self.max_entries
        for key in list(self._entries.keys())[:overflow]:
            self._entries.pop(key, None)
