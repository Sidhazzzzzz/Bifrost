"""
Tests for the Bifrost Batch Processor (LOCAL vs REMOTE).
Uses mocked HTTP responses to test the full pipeline without real API calls.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from app.config import Settings
from app.processor import run_batch, process_single_task, TaskResult
from app.classifier import RouteTarget
from app.client import LLMClient
from app.router import ModelRouter


@pytest.fixture
def sample_tasks():
    return [
        {"task_id": "001", "category": "factual", "prompt": "What is the capital of France?"},
        {"task_id": "002", "category": "sentiment", "prompt": "Is this review positive? 'Great!'"},
    ]


@pytest.fixture
def mock_settings(tmp_path, sample_tasks):
    input_file = tmp_path / "tasks.json"
    input_file.write_text(json.dumps(sample_tasks))

    output_file = tmp_path / "results.json"

    return Settings(
        local_base_url="http://localhost:11434/v1",
        local_model="gemma2:2b",
        remote_base_url="https://api.test.com/v1",
        remote_model="llama-3.1-8b-instant",
        remote_api_key="test-key",
        input_path=str(input_file),
        output_path=str(output_file),
        cache_path=str(tmp_path / "cache.json"),
    )


def _mock_response(text="Test response", prompt_tokens=10, completion_tokens=20):
    return httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
        request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
    )


class TestTaskResult:

    def test_submission_format(self):
        result = TaskResult(
            task_id="001",
            response="Paris",
            model_used="gemma2:2b",
            category="factual",
            routed_to="LOCAL",
            complexity_score=0.2,
        )
        d = result.to_submission_dict()
        assert d == {
            "task_id": "001",
            "response": "Paris",
            "model_used": "gemma2:2b",
        }


class TestBatchProcessing:

    @pytest.mark.asyncio
    async def test_full_batch_run(self, mock_settings):
        with patch("app.client.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=_mock_response())
            mock_instance.aclose = MagicMock()  # httpx client close is async or sync, AsyncClient is async
            
            # Since LLMClient has two AsyncClients, let's mock it
            async def mock_aclose():
                pass
            mock_instance.aclose.side_effect = mock_aclose
            
            MockClient.return_value = mock_instance

            report = await run_batch(mock_settings)

            assert report.total_tasks == 2
            assert report.successful == 2
            assert report.failed == 0

            # Check output file was written
            output = Path(mock_settings.output_path)
            assert output.exists()
            data = json.loads(output.read_text())
            assert len(data) == 2

    @pytest.mark.asyncio
    async def test_duplicate_prompts_are_processed_once(self, tmp_path):
        tasks = [
            {"task_id": "001", "prompt": "Write a Python function to add two numbers."},
            {"task_id": "002", "prompt": "Write a Python function to add two numbers."},
        ]
        input_file = tmp_path / "tasks.json"
        input_file.write_text(json.dumps(tasks))
        output_file = tmp_path / "results.json"
        settings = Settings(
            local_base_url="http://localhost:11434/v1",
            local_model="gemma2:2b",
            remote_base_url="https://api.test.com/v1",
            remote_model="llama-3.1-8b-instant",
            remote_api_key="test-key",
            input_path=str(input_file),
            output_path=str(output_file),
            cache_path=str(tmp_path / "cache.json"),
        )

        with patch("app.client.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=_mock_response("def add(a, b): return a + b"))

            async def mock_aclose():
                pass
            mock_instance.aclose.side_effect = mock_aclose

            MockClient.return_value = mock_instance

            report = await run_batch(settings)

            assert report.total_tasks == 2
            assert report.successful == 2
            assert mock_instance.post.call_count == 1  # Once for chat
            data = json.loads(output_file.read_text())
            assert [item["task_id"] for item in data] == ["001", "002"]


class TestAPIServer:

    def test_health_endpoint(self):
        from app.server import create_app
        from fastapi.testclient import TestClient

        settings = Settings(
            local_base_url="http://localhost:11434/v1",
            remote_base_url="https://api.test.com/v1",
            groq_api_key="test-key",
        )
        app = create_app(settings)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
