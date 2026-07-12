"""
Bifrost Model Router (LOCAL vs REMOTE)
Maps RouteTarget to configured local (Ollama) or remote (Groq) models.
"""

from __future__ import annotations

import re
import json
import threading
from pathlib import Path

from app.classifier import Category, RouteTarget
from app.config import Settings

class ModelRouter:
    def __init__(self, settings: Settings) -> None:
        self.local_model = settings.local_model
        self.remote_model = settings.remote_model
        self.remote_models = self._sort_models_by_size(
            settings.allowed_models or (settings.remote_model,)
        )
        self.stats_file = Path("router_stats.json")
        self._lock = threading.Lock()
        
        self.stats = {
            cat.value: {
                "total": 0,
                "success": 0,
                "local_failures": 0,
                "verification_failures": 0,
                "total_latency": 0.0,
                "total_tokens": 0
            } for cat in Category
        }
        self._load_stats()

    def select_model(
        self,
        target: RouteTarget,
        category: Category | None = None,
        complexity_score: float = 0.0,
    ) -> str:
        """Get the model ID corresponding to the target tier."""
        if target == RouteTarget.LOCAL:
            return self.local_model

        if not self.remote_models:
            return self.remote_model

        high_risk = category in {
            Category.CODE_DEBUG,
            Category.CODE_GEN,
            Category.LOGIC,
            Category.FACTUAL,
            Category.SUMMARIZATION,
        }
        if high_risk:
            return self.remote_models[-1]
        return self.remote_models[0]

    def get_tier_summary(self) -> dict[str, str]:
        return {
            "LOCAL": self.local_model,
            "REMOTE": self.remote_model,
            "REMOTE_OPTIONS": ", ".join(self.remote_models),
        }

    def record_task(
        self,
        category: Category,
        success: bool,
        local_failed: bool = False,
        verification_failed: bool = False,
        latency_ms: float = 0.0,
        total_tokens: int = 0
    ) -> None:
        """Record task statistics to automatically adjust routing confidence."""
        with self._lock:
            cat_stats = self.stats.get(category.value)
            if not cat_stats:
                return
            cat_stats["total"] += 1
            if success:
                cat_stats["success"] += 1
            if local_failed:
                cat_stats["local_failures"] += 1
            if verification_failed:
                cat_stats["verification_failures"] += 1
            cat_stats["total_latency"] += latency_ms
            cat_stats["total_tokens"] += total_tokens
            self._save_stats()

    def get_category_stats(self, category: Category) -> dict[str, float]:
        """Return derived statistics for a category."""
        with self._lock:
            s = self.stats.get(category.value)
            if not s or s["total"] == 0:
                return {"success_rate": 1.0, "avg_latency": 0.0, "failure_rate": 0.0, "total": 0}
            return {
                "success_rate": s["success"] / s["total"],
                "avg_latency": s["total_latency"] / s["total"],
                "failure_rate": (s["local_failures"] + s["verification_failures"]) / s["total"],
                "total": s["total"]
            }

    def _load_stats(self) -> None:
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k in self.stats:
                            self.stats[k].update(v)
            except Exception as e:
                print(f"[Router] Error loading stats: {e}")

    def _save_stats(self) -> None:
        try:
            with open(self.stats_file, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            pass

    @staticmethod
    def _sort_models_by_size(models: tuple[str, ...]) -> list[str]:
        return sorted(dict.fromkeys(models), key=ModelRouter._estimate_size_b)

    @staticmethod
    def _estimate_size_b(model_id: str) -> float:
        lower = model_id.lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*b", lower)
        if match:
            return float(match.group(1))
        if "small" in lower or "mini" in lower or "lite" in lower:
            return 3.0
        if "medium" in lower:
            return 14.0
        if "large" in lower or "xl" in lower:
            return 70.0
        return 999.0
