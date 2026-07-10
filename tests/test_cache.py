from app.cache import PersistentResponseCache, make_cache_key
from app.config import Settings


def test_persistent_cache_exact_and_similar_zero_token(tmp_path):
    settings = Settings()
    cache = PersistentResponseCache(str(tmp_path / "responses.json"), settings)
    response = {
        "response": "Positive",
        "category": "sentiment",
        "tier": "LOCAL",
        "routed_to": "LOCAL",
        "model_used": "bifrost-zero-token:lexicon_sentiment",
        "complexity_score": 0.1,
        "confidence": 0.9,
        "escalated": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
    }

    prompt = "Is this review positive or negative? 'The app is fantastic and fast.'"
    key = make_cache_key(prompt, settings)
    cache.set(key, prompt, "sentiment", response)

    assert cache.get_exact(key)["response"] == "Positive"
    similar = cache.get_similar(
        "Is this review positive or negative? 'The app is fantastic and fast'",
        "sentiment",
    )
    assert similar is not None
    assert similar["total_tokens"] == 0


def test_persistent_cache_does_not_fuzzy_reuse_generated_answers(tmp_path):
    settings = Settings()
    cache = PersistentResponseCache(str(tmp_path / "responses.json"), settings)
    response = {
        "response": "def add(a, b): return a + b",
        "category": "code_generation",
        "tier": "REMOTE",
        "routed_to": "REMOTE",
        "model_used": "llama-3.1-8b-instant",
        "complexity_score": 0.9,
        "confidence": 0.9,
        "escalated": False,
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "total_tokens": 20,
        "latency_ms": 1.0,
    }

    prompt = "Write a tiny Python function add(a, b)."
    cache.set(make_cache_key(prompt, settings), prompt, "code_generation", response)

    assert cache.get_similar(
        "Write a tiny Python function add(a,b)",
        "code_generation",
    ) is None
