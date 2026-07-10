# Bifrost - Intelligent Hybrid AI Routing System

Bifrost is an advanced hybrid AI routing system built for the AMD Developer Hackathon ACT II Track 1. It intelligently routes incoming prompts to either a fast, free local LLM (e.g. `gemma2:2b` via Ollama) or a powerful remote LLM (e.g. `llama-3.1-8b-instant` via Fireworks AI), minimizing cost and latency while maximizing answer accuracy.

## Architecture & Optimizations

Bifrost implements a multi-stage pipeline:

1. **Zero-Token Router:** Immediately answers highly structured, deterministic prompts (simple math, NER, sentiment analysis) using regex and heuristics, completely bypassing LLM inference.
2. **Semantic Cache (Numpy Optimized):** A thread-safe persistent cache that uses exact matching and dense vector embedding cosine similarity (`numpy`) to reuse previous responses, cutting costs by 100% for repeated queries.
3. **Adaptive Hybrid Classifier:** A deep reasoning engine that calculates routing confidence by combining:
   - Topic/category heuristics
   - Prompt length and complexity markers
   - Historical success rates of the model on the category
   - Estimated LLM remote API cost 
4. **Two-Stage Quality Verification:** 
   - *Stage 1 (Local):* Fast heuristic checks to reject empty or weak answers (e.g., "As an AI language model").
   - *Stage 2 (Remote Evaluation):* If the local model confidence was borderline, Bifrost triggers a lightweight remote hallucination and correctness evaluation gate before returning the answer.
5. **Observability Engine:** Structured JSON logging captures routing decisions, latency, complexity scores, and token usage for real-time monitoring.
6. **Thread-Safe Concurrency:** Built on `asyncio`, utilizing `aiofiles` and `asyncio.Lock` for race-condition-free execution in high-concurrency environments (FastAPI).

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed locally.
- Fireworks AI API key.

### Installation

```bash
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Running Locally

Ensure Ollama is running:
```bash
ollama serve
ollama pull gemma2:2b
```

Set your API Key:
```bash
export FIREWORKS_API_KEY="your-api-key"
```

Start the FastAPI Server:
```bash
uvicorn app.server:app --reload
```

## Running Benchmarks

Bifrost includes a comprehensive benchmarking suite that measures Accuracy (F1), Latency (Avg and P95), and Remote API Token Costs.

```bash
python benchmark.py
```
This evaluates the `tests/eval_dataset.json` dataset and generates a `benchmark.json` and a markdown leaderboard report (`leaderboard_report.md`).

## Batch Processing Pipeline

Bifrost can process thousands of tasks asynchronously:
```bash
python -m app.main
```
Reads from `tasks.json` and outputs highly optimized completions to `results.json`.

## Docker Support

```bash
docker build -t bifrost .
docker run -p 8000:8000 -e FIREWORKS_API_KEY="your-api-key" bifrost
```

## Team
Developed for AMD Developer Hackathon ACT II Track 1.
