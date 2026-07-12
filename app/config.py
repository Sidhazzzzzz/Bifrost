"""
Bifrost Configuration Module.

Defaults are friendly for local development, but FIREWORKS_* variables take
priority because Track 1 scoring counts Fireworks tokens.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class Settings:
    # --- Local Base Tier (Ollama) ---
    local_base_url: str = "http://127.0.0.1:11434/v1"
    local_model: str = "gemma2:2b"

    # --- Remote Cloud Tier (Fireworks for submission, Groq fallback for dev) ---
    remote_base_url: str = "https://api.groq.com/openai/v1"
    remote_model: str = "llama-3.1-8b-instant"
    remote_api_key: str = ""
    remote_provider: str = "groq"
    allowed_models: tuple[str, ...] = ()
    groq_api_key: str = ""
    fireworks_api_key: str = ""

    # --- Server & Execution Settings ---
    host: str = "0.0.0.0"
    port: int = 8000
    mode: str = "batch"  # "batch" or "serve"
    max_workers: int = 4

    # --- Batch paths ---
    input_path: str = "/input/tasks.json"
    output_path: str = "/output/results.json"

    # --- Complexity Evaluation Threshold ---
    complexity_threshold: float = 0.8

    # --- Persistent cache ---
    cache_path: str = "/cache/responses.json"
    cache_similarity_threshold: float = 1.0
    cache_max_entries: int = 2000


def load_settings() -> Settings:
    s = Settings()
    
    # Read provider keys. Fireworks wins when present because it is the scorer.
    s.groq_api_key = os.getenv("GROQ_API_KEY", "")
    s.fireworks_api_key = os.getenv("FIREWORKS_API_KEY", "")

    allowed_models_raw = os.getenv("ALLOWED_MODELS", "")
    raw_models = [m.strip() for m in allowed_models_raw.split(",") if m.strip()]
    
    if s.fireworks_api_key or os.getenv("FIREWORKS_BASE_URL") or raw_models:
        # If we are using fireworks, ensure proper prefix to prevent 404s/timeouts
        sanitized = []
        for m in raw_models:
            if not m.startswith("accounts/fireworks/models/"):
                sanitized.append(f"accounts/fireworks/models/{m}")
            else:
                sanitized.append(m)
        s.allowed_models = tuple(sanitized)
    else:
        s.allowed_models = tuple(raw_models)
    
    # Inside Docker, we must bridge to host.docker.internal unless OLLAMA_URL is set
    # Check if we are running in a container
    is_docker = os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER") == "true"
    
    if os.getenv("OLLAMA_URL"):
        s.local_base_url = os.getenv("OLLAMA_URL")
    else:
        s.local_base_url = "http://127.0.0.1:11434/v1"
        
    s.local_model = os.getenv("LOCAL_MODEL", "gemma2:2b")

    if s.fireworks_api_key or os.getenv("FIREWORKS_BASE_URL") or s.allowed_models:
        s.remote_provider = "fireworks"
        s.remote_base_url = os.getenv(
            "FIREWORKS_BASE_URL",
            "https://api.fireworks.ai/inference/v1",
        )
        s.remote_api_key = s.fireworks_api_key
        env_remote = os.getenv("REMOTE_MODEL")
        if env_remote and not env_remote.startswith("accounts/fireworks/models/"):
            env_remote = f"accounts/fireworks/models/{env_remote}"
            
        s.remote_model = env_remote or (
            _smallest_model(s.allowed_models)
            if s.allowed_models
            else "accounts/fireworks/models/llama-v3p1-8b-instruct"
        )
        print(f"[Bifrost] REMOTE configured via Fireworks AI (Model: {s.remote_model})")
    else:
        s.remote_provider = "groq"
        s.remote_base_url = "https://api.groq.com/openai/v1"
        s.remote_api_key = s.groq_api_key
        s.remote_model = os.getenv("REMOTE_MODEL", "llama-3.3-70b-versatile")
        print(f"[Bifrost] WARNING: FIREWORKS_API_KEY/ALLOWED_MODELS missing. Falling back to Groq for REMOTE.")
    
    s.host = os.getenv("HOST", "0.0.0.0")
    s.port = int(os.getenv("PORT", "8000"))
    s.mode = os.getenv("BIFROST_MODE", "batch").lower()
    s.max_workers = int(os.getenv("MAX_WORKERS", "4"))
    s.input_path = os.getenv("INPUT_PATH", "/input/tasks.json")
    s.output_path = os.getenv("OUTPUT_PATH", "/output/results.json")
    
    try:
        s.complexity_threshold = float(os.getenv("COMPLEXITY_THRESHOLD", "0.8"))
    except ValueError:
        s.complexity_threshold = 0.8

    s.cache_path = os.getenv("CACHE_PATH", "/cache/responses.json")
    try:
        s.cache_similarity_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
    except ValueError:
        s.cache_similarity_threshold = 0.92
    s.cache_max_entries = int(os.getenv("CACHE_MAX_ENTRIES", "2000"))
        
    return s


def _smallest_model(models: tuple[str, ...]) -> str:
    return min(models, key=_estimate_size_b)


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
