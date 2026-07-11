"""
Tests for the Bifrost Model Router (LOCAL vs REMOTE).
"""

import pytest
from app.router import ModelRouter
from app.config import Settings
from app.classifier import RouteTarget


class TestModelRouter:

    def test_routing(self):
        settings = Settings(
            local_model="gemma2:2b",
            remote_model="llama-3.1-8b-instant"
        )
        router = ModelRouter(settings)
        assert router.select_model(RouteTarget.LOCAL) == "gemma2:2b"
        assert router.select_model(RouteTarget.REMOTE) == "llama-3.1-8b-instant"
        
        summary = router.get_tier_summary()
        assert summary["LOCAL"] == "gemma2:2b"
        assert summary["REMOTE"] == "llama-3.1-8b-instant"
