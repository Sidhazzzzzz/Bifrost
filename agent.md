# Bifrost - Project Context

## Overview
Bifrost is a hybrid token-efficient LLM routing engine for AMD ACT II Track 1.
It minimizes Fireworks token usage while preserving output accuracy.

The scoring path is:

```text
classify -> deterministic zero-token answer when safe -> local Ollama -> cheapest capable Fireworks model
```

## Architecture
- **Cache**: SHA256 prompt/settings cache for repeated `/v1/chat` calls (`app/cache.py`)
- **Classifier**: Pure Python heuristic engine with 8 task categories (`app/classifier.py`)
- **Zero-token solver**: Conservative deterministic answers for sentiment, simple math, NER, facts, and syllogisms (`app/zero_token.py`)
- **Quality guard**: Conservative weak-answer checks before remote retry (`app/quality.py`)
- **Router**: Parses `ALLOWED_MODELS`, sorts by estimated size, and reserves larger remote models for high-risk tasks (`app/router.py`)
- **Client**: OpenAI-compatible async HTTP for Fireworks, Groq fallback, and Ollama (`app/client.py`)
- **Processor**: Batch pipeline with duplicate prompt reuse (`app/processor.py`)
- **Server**: FastAPI demo with chat, compare, stats, and model metadata endpoints (`app/server.py`)

## Environment Variables
- `FIREWORKS_API_KEY` / `GROQ_API_KEY` - API authentication
- `FIREWORKS_BASE_URL` - Fireworks endpoint for submission
- `ALLOWED_MODELS` - comma-separated remote model IDs available to the router
- `REMOTE_MODEL` - optional remote fallback model
- `LOCAL_MODEL` - optional local Ollama model, default `gemma4:e2b`
- `COMPLEXITY_THRESHOLD` - score cutoff for remote routing, default `0.5`
- `BIFROST_MODE` - `batch` or `serve`
- `OLLAMA_URL` - local Ollama bridge URL

## Modes
- **Batch**: `python -m app.main` - submission mode
- **Serve**: `python -m app.main --serve` - demo server on port 8000
