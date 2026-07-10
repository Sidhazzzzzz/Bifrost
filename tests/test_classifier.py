"""
Tests for the Bifrost Task Classifier (LOCAL vs REMOTE).
"""

import pytest
from app.classifier import Category, RouteTarget, Classification, classify


class TestClassifyComplexity:

    def test_empty_prompt(self):
        r = classify("")
        assert r.category == Category.FACTUAL
        assert r.routed_to == RouteTarget.LOCAL

    def test_sentiment_is_local(self):
        r = classify("Is this text positive or negative: 'I love this.'")
        assert r.category == Category.SENTIMENT
        assert r.routed_to == RouteTarget.LOCAL

    def test_code_generation_is_remote(self):
        r = classify("Write a Python function to sort a list using quicksort")
        assert r.category == Category.CODE_GEN
        assert r.routed_to == RouteTarget.REMOTE

    def test_code_debugging_is_remote(self):
        prompt = (
            "Find the bug in this code:\n"
            "```python\ndef add(a, b):\n    return a - b\n```"
        )
        r = classify(prompt)
        assert r.category == Category.CODE_DEBUG
        assert r.routed_to == RouteTarget.REMOTE

    def test_math_is_remote_if_complex(self):
        # Long prompt with complexity keywords and formulas
        r = classify("Please solve this complex algebra equation: 3x + 5 = 20, then explain the process in detail step-by-step.")
        assert r.category == Category.MATH
        assert r.routed_to == RouteTarget.REMOTE
