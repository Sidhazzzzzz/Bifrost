"""
Bifrost Orchestrator
Unifies the routing, execution, verification, and caching logic.
"""

from __future__ import annotations

import asyncio
from typing import Any
import time

from app.classifier import Category, RouteTarget, classify
from app.client import LLMClient
from app.config import Settings
from app.prompts import build_messages, MAX_TOKENS_HINT
from app.quality import is_weak_answer
from app.router import ModelRouter
from app.zero_token import try_zero_token_answer
from app.cache import PersistentResponseCache, make_cache_key
from app.logger import log_routing_decision, log_task_completion

class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        client: LLMClient,
        router: ModelRouter,
        cache: PersistentResponseCache,
    ) -> None:
        self.settings = settings
        self.client = client
        self.router = router
        self.cache = cache

    async def execute_task(
        self,
        task_id: str,
        prompt: str,
        category_hint: str | None = None,
        force_target: str | None = None,
    ) -> dict[str, Any]:
        """
        Executes a single task through the full Bifrost pipeline:
        1. Cache Check (Exact)
        2. Classification
        3. Cache Check (Semantic)
        4. Zero Token
        5. Execution (with retries and verification)
        6. Cache Update
        """
        start_time = time.perf_counter()
        
        # 1. Exact Cache Check
        cache_key = make_cache_key(prompt, self.settings, force_target=force_target)
        cached = self.cache.get_exact(cache_key)
        if cached:
            cached["latency_ms"] = (time.perf_counter() - start_time) * 1000
            return cached

        # 2. Classification
        classification = classify(prompt, threshold=self.settings.complexity_threshold)
        if category_hint:
            try:
                classification.category = Category(category_hint)
            except ValueError:
                print(f"[Bifrost] WARNING: Invalid category hint '{category_hint}' ignored.")

        # 3. Semantic Cache Check
        similar_cached = self.cache.get_similar(
            prompt,
            classification.category.value,
            force_target=force_target,
            query_embedding=None,
        )
        if similar_cached:
            similar_cached["latency_ms"] = (time.perf_counter() - start_time) * 1000
            return similar_cached

        # Handle explicit override
        target = classification.routed_to
        if force_target:
            try:
                target = RouteTarget(force_target.upper())
            except ValueError:
                pass
                
        # Skip wasted local call for categories that always escalate
        if classification.category in {Category.FACTUAL, Category.LOGIC, Category.SUMMARIZATION}:
            target = RouteTarget.REMOTE
                
        log_routing_decision(
            prompt=prompt,
            category=classification.category.value,
            confidence=getattr(classification, "confidence", 1.0),
            score=classification.complexity_score,
            target=target.value,
            reasoning=getattr(classification, "reasoning", "")
        )

        # 4. Zero Token Layer
        if not force_target:
            zero_answer = try_zero_token_answer(prompt, classification.category)
            if zero_answer:
                response = {
                    "response": zero_answer.response,
                    "category": classification.category.value,
                    "tier": RouteTarget.LOCAL.value,
                    "routed_to": RouteTarget.LOCAL.value,
                    "model_used": f"bifrost-zero-token:{zero_answer.strategy}",
                    "complexity_score": classification.complexity_score,
                    "confidence": getattr(classification, "confidence", 1.0),
                    "reasoning": getattr(classification, "reasoning", "Zero token proof"),
                    "estimated_cost": getattr(classification, "estimated_cost", 0.0),
                    "escalated": False,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": (time.perf_counter() - start_time) * 1000,
                }
                # Safe async cache write
                await self.cache.set_async(
                    cache_key,
                    prompt,
                    classification.category.value,
                    response,
                    embedding=None,
                )
                return response

        # 5. Remote / Local Execution
        max_tokens = MAX_TOKENS_HINT.get(classification.category, 300)
        if classification.category in {Category.FACTUAL, Category.LOGIC}:
            max_tokens = max(max_tokens, 500)
            
        if classification.category in {Category.LOGIC, Category.MATH}:
            max_tokens = max(max_tokens, 600)
            
        model_id = self.router.select_model(
            target,
            classification.category,
            classification.complexity_score,
        )
        messages = build_messages(prompt, classification.category, classification.complexity_score)
        max_tokens = MAX_TOKENS_HINT.get(classification.category, 300)
        
        # Ensure we have enough tokens if we appended CoT
        if classification.category in {Category.FACTUAL, Category.LOGIC}:
            max_tokens = max(max_tokens, 500)

        llm_resp = await self.client.chat(
            messages=messages,
            target=target,
            model=model_id,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        print(f"[Timing] Task {task_id} | {classification.category.value} | Initial {target.value} call took {llm_resp.latency_ms}ms")

        is_good = True
        local_failed = False
        if target == RouteTarget.LOCAL and (llm_resp.error or llm_resp.routed_to == RouteTarget.LOCAL.value):
            if llm_resp.error:
                local_failed = True
                is_good = False
            else:
                # Verify Answer
                is_good = not is_weak_answer(
                    prompt, 
                    llm_resp.text, 
                    classification.category
                )
            
            if not is_good:
                if not local_failed:
                    self.client.stats.total_verification_retries += 1
                    self.router.record_task(classification.category, success=False, verification_failed=True, latency_ms=llm_resp.latency_ms, total_tokens=llm_resp.total_tokens)
                else:
                    self.router.record_task(classification.category, success=False, local_failed=True, latency_ms=llm_resp.latency_ms, total_tokens=llm_resp.total_tokens)
                
                remote_model = self.router.select_model(
                    RouteTarget.REMOTE,
                    classification.category,
                    classification.complexity_score,
                )
                print(f"[Timing] Task {task_id} | {classification.category.value} | Local weak/failed, escalating to REMOTE...")
                # Escalate
                llm_resp = await self.client.chat(
                    messages=messages,
                    target=RouteTarget.REMOTE,
                    model=remote_model,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                print(f"[Timing] Task {task_id} | {classification.category.value} | Escalated REMOTE call took {llm_resp.latency_ms}ms")

        if not llm_resp.error and (is_good or llm_resp.routed_to == RouteTarget.REMOTE.value):
            self.router.record_task(classification.category, success=True, latency_ms=llm_resp.latency_ms, total_tokens=llm_resp.total_tokens)
        elif llm_resp.error and llm_resp.routed_to == RouteTarget.LOCAL.value:
            self.router.record_task(classification.category, success=False, local_failed=True, latency_ms=llm_resp.latency_ms, total_tokens=llm_resp.total_tokens)

        actual_tier = llm_resp.routed_to
        was_escalated = (target.value != actual_tier)
        
        total_latency = (time.perf_counter() - start_time) * 1000
        print(f"[Timing] Task {task_id} | {classification.category.value} | Total processing time: {total_latency:.1f}ms")

        response = {
            "response": llm_resp.text if not llm_resp.error else f"Error: {llm_resp.error}",
            "error": llm_resp.error,
            "category": classification.category.value,
            "tier": actual_tier,
            "routed_to": actual_tier,
            "model_used": llm_resp.model_used,
            "complexity_score": classification.complexity_score,
            "confidence": getattr(classification, "confidence", 0.0),
            "reasoning": getattr(classification, "reasoning", ""),
            "estimated_cost": getattr(classification, "estimated_cost", 0.0),
            "escalated": was_escalated,
            "prompt_tokens": llm_resp.prompt_tokens,
            "completion_tokens": llm_resp.completion_tokens,
            "total_tokens": llm_resp.total_tokens,
            "latency_ms": (time.perf_counter() - start_time) * 1000,
        }

        # 6. Cache Update
        if not response.get("error"):
            await self.cache.set_async(
                cache_key,
                prompt,
                classification.category.value,
                response,
                embedding=None,
            )

        log_task_completion(
            task_id=task_id,
            latency_ms=response["latency_ms"],
            target=actual_tier,
            tokens=response["total_tokens"],
            success=not response.get("error"),
            error=response.get("error")
        )

        return response
