from app.classifier import Category, RouteTarget
from app.config import Settings, load_settings
from app.router import ModelRouter
from app.zero_token import try_zero_token_answer


def test_zero_token_sentiment():
    answer = try_zero_token_answer(
        "Is this review positive or negative? 'The acting was terrible and confusing.'",
        Category.SENTIMENT,
    )
    assert answer is not None
    assert answer.response == "Negative"
    assert answer.confidence >= 0.8


def test_zero_token_linear_math():
    answer = try_zero_token_answer("Solve for x: 2x + 5 = 15", Category.MATH)
    assert answer is not None
    assert "Answer: x = 5" in answer.response


def test_zero_token_ner_extracts_known_entities():
    answer = try_zero_token_answer(
        "Extract all named entities from: 'Elon Musk founded SpaceX in 2002 in Hawthorne, California.'",
        Category.NER,
    )
    assert answer is not None
    assert "Elon Musk" in answer.response
    assert "SpaceX" in answer.response
    assert "Hawthorne, California" in answer.response


def test_router_uses_larger_remote_model_for_code():
    router = ModelRouter(
        Settings(
            remote_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            allowed_models=(
                "accounts/fireworks/models/llama-v3p1-70b-instruct",
                "accounts/fireworks/models/llama-v3p1-8b-instruct",
            ),
        )
    )

    assert router.select_model(
        target=RouteTarget.REMOTE,
        category=Category.FACTUAL,
        complexity_score=0.2,
    ).endswith("8b-instruct")
    assert router.select_model(
        target=RouteTarget.REMOTE,
        category=Category.CODE_GEN,
        complexity_score=0.7,
    ).endswith("70b-instruct")


def test_fireworks_config_uses_smallest_allowed_model_by_default(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv(
        "ALLOWED_MODELS",
        "accounts/fireworks/models/llama-v3p1-70b-instruct,accounts/fireworks/models/llama-v3p1-8b-instruct",
    )
    monkeypatch.delenv("REMOTE_MODEL", raising=False)

    settings = load_settings()

    assert settings.remote_provider == "fireworks"
    assert settings.remote_model.endswith("8b-instruct")
