"""
Bifrost FastAPI Server (LOCAL vs REMOTE)
Interactive demo server with REST API endpoints for live routing
demonstration, analytics, and speed comparison.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.cache import PersistentResponseCache, make_cache_key
from app.classifier import classify, RouteTarget
from app.client import LLMClient
from app.config import Settings, load_settings
from app.prompts import build_messages, MAX_TOKENS_HINT
from app.quality import is_weak_answer
from app.router import ModelRouter
from app.zero_token import try_zero_token_answer
from app.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    force_target: str | None = None


class ChatResponse(BaseModel):
    response: str
    category: str
    tier: str
    routed_to: str
    model_used: str
    complexity_score: float
    confidence: float
    escalated: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


class CompareRequest(BaseModel):
    message: str


class CompareResult(BaseModel):
    tier: str
    routed_to: str
    model_used: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    error: str | None = None


class CompareResponse(BaseModel):
    category: str
    complexity_score: float
    confidence: float
    recommended_tier: str
    results: list[CompareResult]


class StatsResponse(BaseModel):
    total_calls: int
    total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_errors: int
    avg_latency_ms: float
    local_calls: int
    remote_calls: int
    tier_usage: dict[str, int]
    tokens_by_tier: dict[str, int]
    category_usage: dict[str, int]
    cache_entries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_similar_hits: int = 0
    total_verification_retries: int = 0


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = load_settings()

    router = ModelRouter(settings)
    client_holder: dict[str, LLMClient | None] = {"client": None}

    # Analytics counters
    category_usage: dict[str, int] = {}
    tier_usage: dict[str, int] = {"LOCAL": 0, "REMOTE": 0}
    tokens_by_tier: dict[str, int] = {"LOCAL": 0, "REMOTE": 0}
    response_cache = PersistentResponseCache(
        settings.cache_path,
        settings,
        similarity_threshold=settings.cache_similarity_threshold,
        max_entries=settings.cache_max_entries,
    )
    orchestrator_holder: dict[str, Orchestrator | None] = {"orch": None}

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        client_holder["client"] = LLMClient(
            local_base_url=settings.local_base_url,
            remote_base_url=settings.remote_base_url,
            remote_api_key=settings.remote_api_key,
            remote_fallback_model=settings.remote_model,
            timeout=30.0,
        )
        orchestrator_holder["orch"] = Orchestrator(settings, client_holder["client"], router, response_cache)
        print(f"[Bifrost] Server started")
        print(f"[Bifrost] Model routing targets: {router.get_tier_summary()}")
        yield
        if client_holder["client"]:
            await client_holder["client"].close()

    app = FastAPI(
        title="Bifrost",
        description="Hybrid Token-Efficient LLM Routing Engine",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_client() -> LLMClient:
        c = client_holder["client"]
        if c is None:
            raise HTTPException(status_code=503, detail="Client not initialised")
        return c

    def get_orchestrator() -> Orchestrator:
        o = orchestrator_holder["orch"]
        if o is None:
            raise HTTPException(status_code=503, detail="Orchestrator not initialised")
        return o

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def landing_page() -> HTMLResponse:
        index_file = static_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="<h1>Bifrost</h1><p>Frontend not found. Use /v1/chat API.</p>"
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "healthy",
            "local_endpoint": settings.local_base_url,
            "remote_endpoint": settings.remote_base_url,
        }

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        orch = get_orchestrator()
        
        try:
            res = await orch.execute_task(req.message, force_target=req.force_target)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
            
        if res.get("error"):
            raise HTTPException(status_code=502, detail=res["error"])
            
        actual_tier = res.get("tier", "LOCAL")
        cat_key = res.get("category", "unknown")
        
        category_usage[cat_key] = category_usage.get(cat_key, 0) + 1
        tier_usage[actual_tier] = tier_usage.get(actual_tier, 0) + 1
        tokens_by_tier[actual_tier] = tokens_by_tier.get(actual_tier, 0) + res.get("total_tokens", 0)

        return ChatResponse(
            response=res.get("response", ""),
            category=cat_key,
            tier=actual_tier,
            routed_to=res.get("routed_to", actual_tier),
            model_used=res.get("model_used", ""),
            complexity_score=res.get("complexity_score", 0.0),
            confidence=res.get("confidence", 0.0),
            escalated=res.get("escalated", False),
            prompt_tokens=res.get("prompt_tokens", 0),
            completion_tokens=res.get("completion_tokens", 0),
            total_tokens=res.get("total_tokens", 0),
            latency_ms=res.get("latency_ms", 0.0),
        )

    @app.post("/v1/compare", response_model=CompareResponse)
    async def compare(req: CompareRequest) -> CompareResponse:
        client = get_client()

        classification = classify(req.message, threshold=settings.complexity_threshold)
        messages = build_messages(req.message, classification.category)
        max_tokens = MAX_TOKENS_HINT.get(classification.category, 300)

        results: list[CompareResult] = []

        async def call_target(target: RouteTarget) -> CompareResult:
            model_id = router.select_model(
                target,
                classification.category,
                classification.complexity_score,
            )
            resp = await client.chat(
                messages=messages,
                target=target,
                model=model_id,
                max_tokens=max_tokens,
            )
            return CompareResult(
                tier=target.value,
                routed_to=target.value,
                model_used=resp.model_used,
                response=resp.text,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                total_tokens=resp.total_tokens,
                latency_ms=resp.latency_ms,
                error=resp.error,
            )

        target_results = await asyncio.gather(
            call_target(RouteTarget.LOCAL),
            call_target(RouteTarget.REMOTE),
            return_exceptions=True,
        )

        for r in target_results:
            if isinstance(r, CompareResult):
                results.append(r)
            elif isinstance(r, Exception):
                results.append(CompareResult(
                    tier="ERROR",
                    routed_to="ERROR",
                    model_used="unknown",
                    response="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=0.0,
                    error=str(r),
                ))

        return CompareResponse(
            category=classification.category.value,
            complexity_score=classification.complexity_score,
            confidence=getattr(classification, "confidence", 1.0),
            recommended_tier=classification.routed_to.value,
            results=results,
        )

    @app.get("/v1/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        client = get_client()
        s = client.stats
        cache_stats = response_cache.stats()
        return StatsResponse(
            total_calls=s.total_calls,
            total_tokens=s.total_tokens,
            total_prompt_tokens=s.total_prompt_tokens,
            total_completion_tokens=s.total_completion_tokens,
            total_errors=s.total_errors,
            avg_latency_ms=round(s.total_latency_ms / s.total_calls, 1) if s.total_calls > 0 else 0.0,
            local_calls=s.local_calls,
            remote_calls=s.remote_calls,
            tier_usage=dict(tier_usage),
            tokens_by_tier=dict(tokens_by_tier),
            category_usage=dict(category_usage),
            cache_entries=cache_stats["entries"],
            cache_hits=cache_stats["hits"],
            cache_misses=cache_stats["misses"],
            cache_similar_hits=cache_stats["similar_hits"],
            total_verification_retries=s.total_verification_retries,
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        remote_models = router.remote_models or [settings.remote_model]
        return {
            "provider": f"Bifrost Hybrid ({settings.remote_provider})",
            "tier_map": {
                "LOCAL": settings.local_model,
                "REMOTE": settings.remote_model,
            },
            "models": [
                {
                    "model_id": settings.local_model,
                    "estimated_params_b": 2.0,
                    "tier": "LOCAL",
                },
            ] + [
                {
                    "model_id": model_id,
                    "estimated_params_b": ModelRouter._estimate_size_b(model_id),
                    "tier": "REMOTE",
                }
                for model_id in remote_models
            ],
            "complexity_threshold": settings.complexity_threshold,
        }

    return app
