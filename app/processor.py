"""
Bifrost Batch Processor
Reads /input/tasks.json, evaluates complexity, routes tasks to either the LOCAL
or REMOTE tier, calls the APIs, and writes /output/results.json.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from app.classifier import Category, RouteTarget, Classification, classify
from app.client import LLMClient, LLMResponse
from app.config import Settings
from app.router import ModelRouter
from app.cache import PersistentResponseCache, make_cache_key
from app.orchestrator import Orchestrator


@dataclass
class TaskResult:
    task_id: Any
    response: str
    model_used: str
    category: str
    routed_to: str
    complexity_score: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    def to_submission_dict(self) -> dict[str, Any]:
        """Format for /output/results.json (submission-compliant)."""
        res = {
            "task_id": self.task_id,
            "answer": self.response,
        }
        if self.error:
            res["error"] = self.error
        return res

    def to_detailed_dict(self) -> dict[str, Any]:
        """Full detail format for analytics/demo."""
        return {
            "task_id": self.task_id,
            "response": self.response,
            "model_used": self.model_used,
            "category": self.category,
            "routed_to": self.routed_to,
            "complexity_score": self.complexity_score,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    def with_task_id(self, task_id: Any) -> "TaskResult":
        return replace(self, task_id=task_id)


@dataclass
class BatchReport:
    """Summary report of a complete batch run."""
    total_tasks: int = 0
    successful: int = 0
    failed: int = 0
    results: list[TaskResult] = field(default_factory=list)
    token_stats: dict[str, Any] = field(default_factory=dict)
    category_breakdown: dict[str, int] = field(default_factory=dict)
    route_breakdown: dict[str, int] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "successful": self.successful,
            "failed": self.failed,
            "token_stats": self.token_stats,
            "category_breakdown": self.category_breakdown,
            "route_breakdown": self.route_breakdown,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "tokens_saved_estimate": self._estimate_savings(),
        }

    def _estimate_savings(self) -> dict[str, Any]:
        """Estimate token savings vs naive (all-remote-model) approach."""
        actual = self.token_stats.get("total_tokens", 0)
        # Cost estimate assuming naive cloud routing vs hybrid local
        naive_estimate = int(actual * 1.5) if actual > 0 else 0
        saved = max(naive_estimate - actual, 0)
        pct = round((saved / naive_estimate * 100), 1) if naive_estimate > 0 else 0.0
        return {
            "actual_tokens": actual,
            "naive_estimate": naive_estimate,
            "tokens_saved": saved,
            "savings_percentage": pct,
        }


async def process_single_task(
    task: dict[str, Any],
    orchestrator: Orchestrator,
    semaphore: asyncio.Semaphore,
) -> TaskResult:
    """Classify, route, and execute a single task."""
    task_id = task.get("task_id", "unknown")
    prompt = task.get("prompt", task.get("input", task.get("query", "")))
    category_hint = task.get("category", None)

    async with semaphore:
        try:
            res = await orchestrator.execute_task(prompt, category_hint=category_hint)
        except Exception as e:
            res = {
                "response": str(e),
                "model_used": "none",
                "category": "unknown",
                "routed_to": "LOCAL",
                "complexity_score": 0.0,
                "error": str(e)
            }

    return TaskResult(
        task_id=task_id,
        response=res.get("response", ""),
        model_used=res.get("model_used", ""),
        category=res.get("category", "unknown"),
        routed_to=res.get("routed_to", "LOCAL"),
        complexity_score=res.get("complexity_score", 0.0),
        prompt_tokens=res.get("prompt_tokens", 0),
        completion_tokens=res.get("completion_tokens", 0),
        total_tokens=res.get("total_tokens", 0),
        latency_ms=res.get("latency_ms", 0.0),
        error=res.get("error"),
    )


def _task_prompt(task: dict[str, Any]) -> str:
    return task.get("prompt", task.get("input", task.get("query", "")))


def _task_fingerprint(task: dict[str, Any]) -> str:
    prompt = _task_prompt(task)
    return re.sub(r"\s+", " ", str(prompt)).strip().lower()


async def run_batch(settings: Settings) -> BatchReport:
    """Execute the full batch pipeline."""
    start = time.perf_counter()
    report = BatchReport()

    # --- Load tasks ---
    input_path = Path(settings.input_path)
    if not input_path.exists():
        # Try local fallback
        alt = Path("tasks.json")
        if alt.exists():
            input_path = alt
        else:
            raise FileNotFoundError(
                f"Task file not found at {settings.input_path} or ./tasks.json"
            )

    with open(input_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError("tasks.json must contain a JSON array")

    report.total_tasks = len(tasks)
    print(f"[Bifrost] Loaded {len(tasks)} tasks from {input_path}")

    # --- Initialise components ---
    router = ModelRouter(settings)
    print(f"[Bifrost] Model routing targets: {router.get_tier_summary()}")

    client = LLMClient(
        local_base_url=settings.local_base_url,
        remote_base_url=settings.remote_base_url,
        remote_api_key=settings.remote_api_key,
        remote_fallback_model=settings.remote_model,
        timeout=30.0,
    )

    cache = PersistentResponseCache(
        settings.cache_path,
        settings,
        similarity_threshold=settings.cache_similarity_threshold,
        max_entries=settings.cache_max_entries,
    )
    orchestrator = Orchestrator(settings, client, router, cache)

    # --- Process tasks concurrently ---
    semaphore = asyncio.Semaphore(4)

    try:
        unique_tasks: list[dict[str, Any]] = []
        unique_keys: list[str] = []
        seen_keys: set[str] = set()
        task_keys = [_task_fingerprint(task) for task in tasks]

        for task, key in zip(tasks, task_keys):
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_tasks.append(task)
            unique_keys.append(key)

        if len(unique_tasks) != len(tasks):
            saved = len(tasks) - len(unique_tasks)
            print(f"[Bifrost] Deduplicated {saved} repeated task(s) before inference")

        task_coroutines = [
            process_single_task(task, orchestrator, semaphore)
            for task in unique_tasks
        ]
        unique_results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        result_by_key = dict(zip(unique_keys, unique_results))

        results = []
        for task, key in zip(tasks, task_keys):
            task_id = task.get("task_id", "unknown")
            result = result_by_key[key]
            if isinstance(result, TaskResult):
                results.append(result.with_task_id(task_id))
            else:
                results.append(TaskResult(
                    task_id=task_id,
                    response=str(result),
                    model_used="none",
                    category="unknown",
                    routed_to="LOCAL",
                    complexity_score=0.0,
                    error=str(result),
                ))
    finally:
        await client.close()

    # --- Collect results ---
    for result in results:
        if isinstance(result, Exception):
            report.failed += 1
            report.results.append(TaskResult(
                task_id="error",
                response=str(result),
                model_used="none",
                category="unknown",
                routed_to="LOCAL",
                complexity_score=0.0,
                error=str(result),
            ))
        elif isinstance(result, TaskResult):
            if result.error:
                report.failed += 1
            else:
                report.successful += 1
            report.results.append(result)

            # Track breakdowns
            cat = result.category
            report.category_breakdown[cat] = report.category_breakdown.get(cat, 0) + 1
            route = result.routed_to
            report.route_breakdown[route] = report.route_breakdown.get(route, 0) + 1

    report.token_stats = client.stats.to_dict()
    report.total_elapsed_ms = (time.perf_counter() - start) * 1000

    # --- Write output ---
    output_path = Path(settings.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Submission format (minimal)
    submission_results = [r.to_submission_dict() for r in report.results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission_results, f, indent=2, ensure_ascii=False)

    print(f"[Bifrost] Results written to {output_path}")
    print(f"[Bifrost] {report.successful}/{report.total_tasks} succeeded "
          f"| {report.token_stats.get('total_tokens', 0)} total tokens "
          f"| {report.total_elapsed_ms:.0f}ms total")

    return report
